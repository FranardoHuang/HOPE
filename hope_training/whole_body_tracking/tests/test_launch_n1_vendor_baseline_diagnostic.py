from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_n1_vendor_baseline_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location("n1_vendor_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
L = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(L)
_REAL_LOOP_POLICY_PIN = L.BH_LOOP_C_BASE_POLICY_CONTRACT_SHA256
_REAL_BLOCK_POLICY_PIN = L.BH_BLOCK_BASE_POLICY_CONTRACT_SHA256
_REAL_STATIC_REWARD_PIN = L.STATIC_EFFECTIVE_REWARD_RECIPE_SHA256
_REAL_ADAPTIVE_REWARD_PIN = (
    L.MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256
)
_REAL_LOOP_CONFIG = L._LOOP_ACTION_CONFIG
if _REAL_LOOP_CONFIG.runtime_contract.sha256 is None:
    # The production registry deliberately enters a sha256=None epoch while
    # corrected artifacts are being minted.  Unit tests still exercise every
    # launcher invariant against the last tracked, internally consistent
    # fixture; production code remains unmodified and fail-closed.  This
    # historical pin does not authorize or assert the current push interval
    # or integrated-probe stage contract.
    _TEST_LOOP_CONFIG = replace(
        _REAL_LOOP_CONFIG,
        required_identity_manifest=L._R.ArtifactPin(
            "configs/a3_vendor_runtime_contract_20260731/required_identity.v1.json",
            "3b2c5992d673b0be3ca4e7c27f1c4d0cdbfd2b87b6d3c6f6387fb9ea401904af",
        ),
        runtime_contract=L._R.ArtifactPin(
            "configs/a3_vendor_runtime_authority_20260731/"
            "bh_loop_c.shared_ready.training_contract.json",
            "38974f1bc5da8140aec24e07dd2d59d9b7cc90ed52acdd20f54564dd70368fba",
        ),
        runtime_authority_receipt=L._R.ArtifactPin(
            "configs/a3_vendor_runtime_authority_20260731/"
            "bh_loop_c.vendor_runtime_authority.v1.json",
            "0cc33f12a2d71d1ad61175a41c357b5e43cad00a32d04fd1abc42ac61a91bc41",
        ),
        contact_bundle=L._R.ArtifactPin(
            "configs/n1_contact_vendor_a3_20260731_r3/"
            "bh_loop_c.bundle.v2.72905f53af87.json",
            "72905f53af87b3d17dee30777a8e24cf3e1e97cc26118bd4b36f4da20d86a466",
        ),
        fixed_domain_initial_receipt=L._R.ArtifactPin(
            "configs/test-loop-fixed-domain.json", "5" * 64
        ),
        reward_economy_receipt=L._R.ArtifactPin(
            "configs/test-reward-economy.json", "6" * 64
        ),
    )
else:
    _TEST_LOOP_CONFIG = _REAL_LOOP_CONFIG

# Downstream receipt layers materialize at different clean-successor epochs.
# Keep the synthetic launcher fixture internally complete so a negative test
# can remove exactly one requested pin without an earlier, unrelated planned
# layer intercepting the assertion.  Production registry pins remain untouched.
if _TEST_LOOP_CONFIG.fixed_domain_initial_receipt.sha256 is None:
    _TEST_LOOP_CONFIG = replace(
        _TEST_LOOP_CONFIG,
        fixed_domain_initial_receipt=L._R.ArtifactPin(
            "configs/test-loop-fixed-domain.json", "5" * 64
        ),
    )
if _TEST_LOOP_CONFIG.reward_economy_receipt.sha256 is None:
    _TEST_LOOP_CONFIG = replace(
        _TEST_LOOP_CONFIG,
        reward_economy_receipt=L._R.ArtifactPin(
            "configs/test-reward-economy.json", "6" * 64
        ),
    )
VENDOR_CONTRACT_SHA = _TEST_LOOP_CONFIG.runtime_contract.sha256
assert VENDOR_CONTRACT_SHA is not None


def test_r9_launcher_starts_policy_closed_but_preserves_reward_bytes() -> None:
    assert _REAL_LOOP_POLICY_PIN is None
    assert _REAL_BLOCK_POLICY_PIN is None
    assert _REAL_STATIC_REWARD_PIN == (
        "845d75b4f409725e9dfc7070b1070a6dd6385486c79a6c5a1aec60c41c42ff02"
    )
    assert _REAL_ADAPTIVE_REWARD_PIN == (
        "ce910ac22d5b226b849b16b3a3cd2b633d86207a489d3188222d86c835715803"
    )


def _canonical(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize("stage", ("smoke", "long"))
def test_vendor_economy_gate_is_absent_outside_probe(stage):
    assert L._vendor_probe_exec_environment(
        {"stage": stage, "diagnostic_update_profile": False}
    ) == {}


def test_vendor_economy_gate_is_claim_owned_for_probe_only():
    spec = {"stage": "probe", "diagnostic_update_profile": False}
    assert L._vendor_probe_exec_environment(spec) == {
        L.REWARD_PPO_ECONOMY_GATE_ENV: "1"
    }
    assert L._B._diagnostic_update_profile_environment(spec) == {
        L.REWARD_PPO_ECONOMY_GATE_ENV: "1"
    }


def test_vendor_economy_gate_refuses_update_profile_or_unknown_stage():
    with pytest.raises(L.LaunchRefused, match="exactly false"):
        L._vendor_probe_exec_environment(
            {"stage": "probe", "diagnostic_update_profile": True}
        )
    with pytest.raises(L.LaunchRefused, match="unknown stage"):
        L._vendor_probe_exec_environment(
            {"stage": "other", "diagnostic_update_profile": False}
        )


def _spec(
    tmp_path: Path,
    *,
    seed: int,
    stage: str,
    sigma_profile: str = L.STATIC_SIGMA_PROFILE,
) -> dict:
    checkout = tmp_path / "checkout"
    checkout.mkdir(exist_ok=True)
    isaac_python = tmp_path / "python.sh"
    isaac_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    isaac_python.chmod(0o755)
    namespace_parent = tmp_path / "runs"
    namespace_parent.mkdir(exist_ok=True)
    suffix = (
        f"-{sigma_profile}"
        if sigma_profile != L.STATIC_SIGMA_PROFILE
        else ""
    )
    lane_id = (
        L.LOOP_ADAPTIVE_LANE
        if sigma_profile == L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE
        else L.LOOP_STATIC_LANE
    )
    namespace = namespace_parent / f"vendor-{lane_id}-seed{seed}-{stage}{suffix}"
    budget = {
        "smoke": (1, 2, 1),
        "probe": (4096, 5, 1),
        # Historical invalid-input fixture only.  The live launcher contract
        # must reject this retired standalone stage.
        "push_evidence": (4096, 32, 8),
        "long": (4096, 20_001, 100),
    }[stage]
    policy_sha = "c" * 64
    return {
        "schema_version": 1,
        "kind": L.SPEC_KIND,
        "source": {
            "checkout": str(checkout),
            "commit_sha": "a" * 40,
            "isaac_python": str(isaac_python),
        },
        L.VENDOR_LANE_FIELD: lane_id,
        "action_id": "bh_loop_c",
        "scope": "upper",
        "bundle": dict(L.CANONICAL_BUNDLE_PIN),
        L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD: dict(
            L._R.require_materialized_pin(
                _TEST_LOOP_CONFIG.fixed_domain_initial_receipt,
                action_id="bh_loop_c",
                layer="fixed-domain initial receipt",
            )
        ),
        L.REWARD_ECONOMY_RECEIPT_FIELD: dict(
            L._R.require_materialized_pin(
                _TEST_LOOP_CONFIG.reward_economy_receipt,
                action_id="bh_loop_c",
                layer="reward economy receipt",
            )
        ),
        "policy_contract_sha256": policy_sha,
        "reward_profile": L.REWARD_PROFILE,
        L.SIGMA_PROFILE_FIELD: sigma_profile,
        L.SIGMA_VARIANT_IDENTITY_FIELD: (
            L._sigma_variant_scientific_identity_sha256(
                policy_sha, sigma_profile
            )
        ),
        "expected_effective_reward_recipe_sha256": (
            L.STATIC_EFFECTIVE_REWARD_RECIPE_SHA256
        ),
        L.VENDOR_CONTRACT_FIELD: VENDOR_CONTRACT_SHA,
        "seed": seed,
        "stage": stage,
        "num_envs": budget[0],
        "max_iterations": budget[1],
        "save_interval": budget[2],
        "diagnostic_update_profile": False,
        "gpu": {
            "index": seed,
            "uuid": f"GPU-vendor-seed-{seed}",
            "owner": "Franco",
            "lock_path": f"/tmp/hope_lean_queue_gpu{seed}.lock",
            "require_empty": True,
        },
        "namespace": str(namespace),
        "log_path": str(namespace / "run.log"),
    }


@pytest.fixture(autouse=True)
def _materialized_lane_policy_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_get_action_config = L._R.get_action_config

    def test_get_action_config(action_id):
        if action_id == "bh_loop_c":
            return _TEST_LOOP_CONFIG
        if action_id == "bh_block":
            return _materialized_block_config()
        return original_get_action_config(action_id)

    monkeypatch.setattr(L._R, "get_action_config", test_get_action_config)
    monkeypatch.setattr(L, "_LOOP_ACTION_CONFIG", _TEST_LOOP_CONFIG)
    monkeypatch.setattr(
        L,
        "VENDOR_IDENTITY_MANIFEST_SOURCE",
        _TEST_LOOP_CONFIG.required_identity_manifest.path,
    )
    monkeypatch.setattr(
        L,
        "VENDOR_IDENTITY_MANIFEST_SHA256",
        _TEST_LOOP_CONFIG.required_identity_manifest.sha256,
    )
    monkeypatch.setattr(
        L,
        "VENDOR_AUTHORITY_RECEIPT_SHA256",
        _TEST_LOOP_CONFIG.runtime_authority_receipt.sha256,
    )
    monkeypatch.setattr(
        L,
        "CANONICAL_BUNDLE_PIN",
        {
            "path": _TEST_LOOP_CONFIG.contact_bundle.path,
            "sha256": _TEST_LOOP_CONFIG.contact_bundle.sha256,
        },
    )
    monkeypatch.setattr(
        L, "BH_LOOP_C_BASE_POLICY_CONTRACT_SHA256", "c" * 64
    )
    monkeypatch.setattr(
        L, "BH_BLOCK_BASE_POLICY_CONTRACT_SHA256", "c" * 64
    )
    monkeypatch.setattr(
        L, "STATIC_EFFECTIVE_REWARD_RECIPE_SHA256", "d" * 64
    )
    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        "e" * 64,
    )


def _bundle() -> dict:
    return {
        "motion": {"path": "motions/bh_loop_c_upper.npz"},
        "manifest": {"path": "manifest.json", "sha256": "e" * 64},
        "dynamic_ready": {
            "artifact": {"path": "ready.json", "sha256": "f" * 64},
            "nominal_hold_receipt": {
                "path": "hold.json",
                "sha256": "0" * 64,
            },
        },
    }


def _select_block_lane(document: dict) -> None:
    block = L._action_config("bh_block")
    document[L.VENDOR_LANE_FIELD] = L.BLOCK_STATIC_LANE
    document["action_id"] = "bh_block"
    document["bundle"] = dict(
        L._R.require_materialized_pin(
            block.contact_bundle,
            action_id=block.action_id,
            layer="contact bundle",
        )
    )
    document[L.VENDOR_CONTRACT_FIELD] = block.runtime_contract.sha256
    document[L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD] = dict(
        L._R.require_materialized_pin(
            block.fixed_domain_initial_receipt,
            action_id=block.action_id,
            layer="fixed-domain initial receipt",
        )
    )
    document[L.REWARD_ECONOMY_RECEIPT_FIELD] = dict(
        L._R.require_materialized_pin(
            block.reward_economy_receipt,
            action_id=block.action_id,
            layer="reward economy receipt",
        )
    )
    namespace = Path(document["namespace"])
    name = namespace.name.replace(L.LOOP_STATIC_LANE, L.BLOCK_STATIC_LANE)
    document["namespace"] = str(namespace.with_name(name))
    document["log_path"] = str(Path(document["namespace"]) / "run.log")


def _loop_dynamic_artifact(*, contract_sha: str | None) -> dict:
    sources = {
        "stable_motion": {
            **dict(L._R.stable_pin(L._LOOP_ACTION_CONFIG.stable_motion)),
            "frame_index": 0,
        }
    }
    if contract_sha is not None:
        sources["runtime_training_contract"] = {"sha256": contract_sha}
    return {"action_id": "bh_loop_c", "sources": sources}


def _materialized_block_config():
    block = L._R.ACTION_CONFIGS["bh_block"]
    return replace(
        block,
        required_identity_manifest=L._R.ArtifactPin(
            "configs/block-required-identity.json", "1" * 64
        ),
        runtime_contract=L._R.ArtifactPin(
            block.runtime_contract.path, "2" * 64
        ),
        runtime_authority_receipt=L._R.ArtifactPin(
            block.runtime_authority_receipt.path, "3" * 64
        ),
        contact_bundle=L._R.ArtifactPin(
            "configs/block-bundle.json", "4" * 64
        ),
        fixed_domain_initial_receipt=L._R.ArtifactPin(
            "configs/block-fixed-domain.json", "5" * 64
        ),
        reward_economy_receipt=L._R.ArtifactPin(
            "configs/test-reward-economy.json", "6" * 64
        ),
    )


def _prelaunch_receipt_documents() -> tuple[dict, dict]:
    loop = L._action_config("bh_loop_c")
    block = L._action_config("bh_block")
    fixed_content = {
        "action_id": "bh_loop_c",
        "scope": "upper",
        "domain_epoch": 0,
        "planned_output_path": loop.fixed_domain_initial_receipt.path,
        "authorized_lane_ids": [
            L.LOOP_STATIC_LANE,
            L.LOOP_ADAPTIVE_LANE,
        ],
        "registry_action_source_identity": L._R.action_source_identity(loop),
        "registry_action_source_identity_sha256": (
            L._R.action_source_identity_sha256(loop)
        ),
    }
    fixed = {
        "schema_version": 1,
        "kind": "n1_fixed_domain_initial_receipt_v1",
        "content": fixed_content,
        "content_sha256": "7" * 64,
    }
    economy = {
        "schema_version": 1,
        "kind": "action_ball_reward_ppo_economy_receipt_v1",
        "authorization": {
            "diagnostic_unauthorized": True,
            "training": False,
            "resume": False,
            "promotion": False,
            "export": False,
            "judge": False,
            "deployment": False,
            "hardware": False,
        },
        "sources": {
            "registry_action_source_identities": [
                {
                    "action_id": config.action_id,
                    "identity": L._R.action_source_identity(config),
                    "sha256": L._R.action_source_identity_sha256(config),
                }
                for config in (loop, block)
            ],
            "runtime_training_contracts": [
                {
                    "action_id": config.action_id,
                    "path": config.runtime_contract.path,
                    "sha256": config.runtime_contract.sha256,
                }
                for config in (loop, block)
            ],
            "registry_output": {
                "path": loop.reward_economy_receipt.path,
            },
        },
        "reward_economy": {
            "reward_global_scalar": 1.0,
            "policy_step_dt_s": 0.02,
            "effective_reward_recipe_sha256": "d" * 64,
        },
        "ppo_economy": {
            "required_final_policy": {
                "fresh_only": True,
                "resume_from_scalar_checkpoint_prohibited": True,
                "noise_std_type": "log",
                "parameter_name": "log_std",
                "init_config_sigma": 0.02,
                "required_realized_init_noise_std": 0.02,
                "strictly_positive_by_construction": True,
            }
        },
        "runtime_4096x5_telemetry_consumer": {
            "status": "wired_probe_gate_runtime_evidence_required",
        },
        "content_sha256": "8" * 64,
    }
    return fixed, economy


def _install_prelaunch_receipt_fakes(
    monkeypatch: pytest.MonkeyPatch,
    fixed: dict,
    economy: dict,
) -> None:
    def load(_checkout, _commit, pin, *, name):
        if "fixed-domain" in name:
            return dict(pin), copy.deepcopy(fixed)
        return dict(pin), copy.deepcopy(economy)

    fixed_module = SimpleNamespace(
        validate_receipt_document=lambda document: document,
    )
    economy_module = SimpleNamespace(
        EXPECTED_EFFECTIVE_REWARD_SHA256="d" * 64,
        validate_receipt_document=lambda document: document,
    )
    monkeypatch.setattr(L._B, "_load_tracked_json", load)
    monkeypatch.setattr(
        L,
        "_load_tracked_receipt_validator",
        lambda _checkout, _commit, path, **_kwargs: (
            fixed_module
            if path == L.FIXED_DOMAIN_RECEIPT_PRODUCER_SOURCE
            else economy_module
        ),
    )


def test_legacy_loop_bundle_alias_is_empty_during_materialization_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        L._LOOP_ACTION_CONFIG,
        contact_bundle=L._R.ArtifactPin("configs/planned.json", None),
    )
    monkeypatch.setattr(L, "_LOOP_ACTION_CONFIG", config)
    assert dict(L._legacy_loop_bundle_alias()) == {}
    with pytest.raises(TypeError):
        L._legacy_loop_bundle_alias()["path"] = "operator.json"


def _explicit_identity_groups(checkout: Path) -> list[dict]:
    names = list(L._load_vendor_authority_module(checkout).RUNTIME_JOINT_NAMES)
    groups = []
    cursor = 0
    for size in [3] * 7 + [2] * 5:
        groups.append(
            {
                "joints": names[cursor : cursor + size],
                "stiffness": 1.0,
                "damping": 1.0,
                "effort_limit": 1.0,
                "armature": 1.0,
                "action_scale": 0.25,
            }
        )
        cursor += size
    assert cursor == 31 and len(groups) == 12
    return groups


@pytest.mark.parametrize("stage", ["smoke", "probe"])
def test_exact_seed_and_stage_namespaces_are_accepted(
    tmp_path: Path, stage: str
) -> None:
    seed = 0
    normalized = L._validate_spec_document(_spec(tmp_path, seed=seed, stage=stage))
    assert normalized["seed"] == seed
    assert normalized["stage"] == stage
    assert Path(normalized["namespace"]).name == (
        f"vendor-{L.LOOP_STATIC_LANE}-seed{seed}-{stage}"
    )
    assert normalized[L.SIGMA_PROFILE_FIELD] == L.STATIC_SIGMA_PROFILE
    assert (
        normalized[L.SIGMA_VARIANT_IDENTITY_FIELD]
        == normalized["policy_contract_sha256"]
    )
    assert normalized[L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD] == dict(
        L._R.require_materialized_pin(
            _TEST_LOOP_CONFIG.fixed_domain_initial_receipt,
            action_id="bh_loop_c",
            layer="fixed-domain initial receipt",
        )
    )
    assert normalized[L.REWARD_ECONOMY_RECEIPT_FIELD] == dict(
        L._R.require_materialized_pin(
            _TEST_LOOP_CONFIG.reward_economy_receipt,
            action_id="bh_loop_c",
            layer="reward economy receipt",
        )
    )


def test_missing_sigma_profile_is_refused_by_lane_contract(
    tmp_path: Path,
) -> None:
    document = _spec(tmp_path, seed=0, stage="smoke")
    del document[L.SIGMA_PROFILE_FIELD]
    with pytest.raises(L.LaunchRefused, match="requires its contract, lane"):
        L._validate_spec_document(document)


def test_other_seed_and_non_exact_stage_are_refused(tmp_path: Path) -> None:
    bad_seed = _spec(tmp_path, seed=0, stage="smoke")
    bad_seed["seed"] = 3
    with pytest.raises(L.LaunchRefused, match="lane seed"):
        L._validate_spec_document(bad_seed)

    bad_budget = _spec(tmp_path, seed=0, stage="probe")
    bad_budget["max_iterations"] = 6
    with pytest.raises(L.LaunchRefused, match="exactly 4096 envs / 5 updates"):
        L._validate_spec_document(bad_budget)

    bad_stage = _spec(tmp_path, seed=0, stage="smoke")
    bad_stage["stage"] = "canary"
    with pytest.raises(L.LaunchRefused, match="stage must be"):
        L._validate_spec_document(bad_stage)

    retired_push_stage = _spec(tmp_path, seed=0, stage="push_evidence")
    with pytest.raises(L.LaunchRefused, match="stage must be"):
        L._validate_spec_document(retired_push_stage)

    stale_bundle = _spec(tmp_path, seed=0, stage="smoke")
    stale_bundle["bundle"] = {"path": "bundle.json", "sha256": "b" * 64}
    with pytest.raises(L.LaunchRefused, match="code-owned canonical pin"):
        L._validate_spec_document(stale_bundle)

    long_stage = _spec(tmp_path, seed=0, stage="long")
    with pytest.raises(L.LaunchRefused, match="vendor_probe_gate_receipt"):
        L._validate_spec_document(long_stage)

    missing_contract = _spec(tmp_path, seed=0, stage="smoke")
    del missing_contract[L.VENDOR_CONTRACT_FIELD]
    with pytest.raises(L.LaunchRefused, match="requires"):
        L._validate_spec_document(missing_contract)

    for field in (
        L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD,
        L.REWARD_ECONOMY_RECEIPT_FIELD,
    ):
        missing_pin = _spec(tmp_path, seed=0, stage="smoke")
        del missing_pin[field]
        with pytest.raises(L.LaunchRefused, match="requires its contract"):
            L._validate_spec_document(missing_pin)

        wrong_pin = _spec(tmp_path, seed=0, stage="smoke")
        wrong_pin[field]["sha256"] = "0" * 64
        with pytest.raises(L.LaunchRefused, match="action-specific code-owned pin"):
            L._validate_spec_document(wrong_pin)

    block_action = _spec(tmp_path, seed=0, stage="smoke")
    _select_block_lane(block_action)
    block_action["bundle"] = dict(L.CANONICAL_BUNDLE_PIN)
    with pytest.raises(L.LaunchRefused, match="action-specific"):
        L._validate_spec_document(block_action)

    unknown_action = _spec(tmp_path, seed=0, stage="smoke")
    unknown_action["action_id"] = "operator_action"
    with pytest.raises(L.LaunchRefused, match="must be one of"):
        L._validate_spec_document(unknown_action)


def test_block_registry_materialized_pins_validate_and_cross_action_tamper_fails_closed(
    tmp_path: Path,
) -> None:
    block = L._action_config("bh_block")
    assert dict(L._R.stable_pin(block.stable_motion)) == {
        "path": (
            "assets/motions/fivebind_20260727/"
            "bh_block_upper_stable_v2.npz"
        ),
        "sha256": (
            "cc9bbccd1b5b6207a0ce9677944ba27fa4a062a1eaa61886d802c9d21830caa0"
        ),
    }
    document = _spec(tmp_path, seed=0, stage="smoke")
    _select_block_lane(document)
    normalized = L._validate_spec_document(document)
    assert normalized["action_id"] == "bh_block"
    assert normalized["bundle"] == dict(
        L._R.require_materialized_pin(
            block.contact_bundle,
            action_id=block.action_id,
            layer="contact bundle",
        )
    )
    assert normalized[L.VENDOR_CONTRACT_FIELD] == block.runtime_contract.sha256

    cross_action_bundle = copy.deepcopy(document)
    cross_action_bundle["bundle"] = dict(L.CANONICAL_BUNDLE_PIN)
    with pytest.raises(L.LaunchRefused, match="action-specific"):
        L._validate_spec_document(cross_action_bundle)

    tampered_contract = copy.deepcopy(document)
    tampered_contract[L.VENDOR_CONTRACT_FIELD] = "0" * 64
    with pytest.raises(L.LaunchRefused, match="action-specific"):
        L._validate_spec_document(tampered_contract)


def test_registry_stable_source_pins_match_tracked_files_for_both_actions() -> None:
    checkout = Path(__file__).resolve().parents[3]
    for action_id in sorted(L._R.ALLOWED_ACTION_IDS):
        config = L._R.get_action_config(action_id)
        for pin in (
            config.stable_motion,
            config.stable_source_manifest,
            config.stable_source_prototype,
        ):
            normalized = dict(L._R.stable_pin(pin))
            assert L._B.sha256_file(checkout / normalized["path"]) == normalized[
                "sha256"
            ]


def test_future_materialized_block_spec_accepts_only_its_own_code_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = _materialized_block_config()
    original = L._action_config
    monkeypatch.setattr(
        L,
        "_action_config",
        lambda action_id: block if action_id == "bh_block" else original(action_id),
    )
    document = _spec(tmp_path, seed=0, stage="smoke")
    _select_block_lane(document)
    document["bundle"] = dict(
        L._R.require_materialized_pin(
            block.contact_bundle,
            action_id=block.action_id,
            layer="contact bundle",
        )
    )
    document[L.VENDOR_CONTRACT_FIELD] = block.runtime_contract.sha256
    normalized = L._validate_spec_document(document)
    assert normalized["action_id"] == "bh_block"

    document["bundle"] = dict(L.CANONICAL_BUNDLE_PIN)
    with pytest.raises(L.LaunchRefused, match="action-specific"):
        L._validate_spec_document(document)


def test_long_is_fail_closed_without_receipt_and_exact_with_pin(
    tmp_path: Path,
) -> None:
    missing = _spec(tmp_path, seed=0, stage="long")
    with pytest.raises(L.LaunchRefused, match="vendor_probe_gate_receipt"):
        L._validate_spec_document(missing)

    admitted = _spec(tmp_path, seed=0, stage="long")
    admitted[L.VENDOR_PROBE_GATE_FIELD] = {
        "path": "configs/n1_vendor_probe_gate/pass.json",
        "sha256": "1" * 64,
    }
    normalized = L._validate_spec_document(admitted)
    assert normalized["stage"] == "long"
    assert normalized[L.VENDOR_PROBE_GATE_FIELD] == admitted[
        L.VENDOR_PROBE_GATE_FIELD
    ]

    non_long = _spec(tmp_path, seed=0, stage="probe")
    non_long[L.VENDOR_PROBE_GATE_FIELD] = admitted[L.VENDOR_PROBE_GATE_FIELD]
    with pytest.raises(L.LaunchRefused, match="permitted only"):
        L._validate_spec_document(non_long)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_envs", 4095),
        ("max_iterations", 4),
        ("max_iterations", 6),
        ("save_interval", 2),
    ],
)
def test_integrated_probe_budget_is_exact(
    tmp_path: Path, field: str, value: int
) -> None:
    spec = _spec(tmp_path, seed=0, stage="probe")
    spec[field] = value

    with pytest.raises(L.LaunchRefused, match="exactly 4096 envs / 5 updates"):
        L._validate_spec_document(spec)


def test_standalone_push_stage_is_absent_from_live_contract() -> None:
    assert "push_evidence" not in L.ALLOWED_STAGES
    assert "push_evidence" not in L.EXACT_STAGE_BUDGETS


def test_argv_selects_vendor_profile_and_forces_stable_ready_plant(
    tmp_path: Path,
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))
    argv = L._build_training_argv(spec, _bundle())

    assert f"task={L.TASK_PROFILE_ID}" in argv
    assert "task=HOPEPingPongActionBall" not in argv
    assert "task.racket.action_ball_diagnostic_unauthorized=true" in argv
    assert "algo.policy.init_noise_std=0.02" in argv
    assert argv.count(L.VENDOR_POLICY_NOISE_STD_OVERRIDE) == 1
    assert argv.count(L.STABLE_READY_PLANT_OVERRIDE) == 1
    assert L.TABLE_ATTRIBUTION_PROBE_OVERRIDE not in argv
    assert argv.count(L.VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX + "smoke") == 1
    assert argv.count(L.VENDOR_CONTRACT_ARG_PREFIX + VENDOR_CONTRACT_SHA) == 1
    assert not any(
        item in L.MONOTONIC_FRESH_CANARY_OVERRIDES
        or item.startswith(L.SIGMA_PROFILE_ARG_PREFIX)
        for item in argv
    )
    forbidden = (
        "push.enable",
        "push_robot",
        "randomize_pd_gains",
        "kp_gain_range",
        "kd_gain_range",
        "control_step_action_delay_",
        "task.rewards.motion_scale=",
        "task.rewards.racket_position_weight=",
        "task.rewards.racket_velocity_weight=",
        "task.rewards.racket_normal_weight=",
    )
    assert not any(fragment in item for item in argv for fragment in forbidden)


@pytest.mark.parametrize(
    ("stage", "expected_count"),
    [("smoke", 0), ("probe", 1)],
)
def test_table_attribution_is_code_owned_probe_only(
    tmp_path: Path, stage: str, expected_count: int
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage=stage))
    argv = L._build_training_argv(spec, _bundle())
    assert argv.count(L.TABLE_ATTRIBUTION_PROBE_OVERRIDE) == expected_count
    assert "task.table_contact_attribution_diagnostic=true" not in argv


def test_table_attribution_refuses_inherited_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="probe"))
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited.append("task.table_contact_attribution_diagnostic=false")
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: inherited
    )
    with pytest.raises(L.LaunchRefused, match="code-owned table attribution"):
        L._build_training_argv(spec, _bundle())


def test_table_attribution_refuses_inherited_append_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="probe"))
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited.append("+task.table_contact_attribution_diagnostic=false")
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: inherited
    )
    with pytest.raises(L.LaunchRefused, match="code-owned table attribution"):
        L._build_training_argv(spec, _bundle())


def test_vendor_log_std_refuses_inherited_override_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited.append("algo.policy.noise_std_type=scalar")
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: inherited
    )
    with pytest.raises(L.LaunchRefused, match="code-owned vendor log_std"):
        L._build_training_argv(spec, _bundle())


def test_every_vendor_lane_is_fresh_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited.append("checkpoint_path=/tmp/legacy-scalar.pt")
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: inherited
    )
    with pytest.raises(L.LaunchRefused, match="fresh vendor log_std"):
        L._build_training_argv(spec, _bundle())


def test_monotonic_sigma_canary_has_one_exact_fresh_only_scientific_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary_reward_sha = "e" * 64
    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        canary_reward_sha,
    )
    document = _spec(
        tmp_path,
        seed=0,
        stage="probe",
        sigma_profile=L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE,
    )
    document["expected_effective_reward_recipe_sha256"] = canary_reward_sha
    spec = L._validate_spec_document(document)
    argv = L._build_training_argv(spec, _bundle())

    assert spec["action_id"] == "bh_loop_c"
    assert L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE in Path(
        spec["namespace"]
    ).name
    assert spec[L.SIGMA_VARIANT_IDENTITY_FIELD] != spec[
        "policy_contract_sha256"
    ]
    assert (
        "task.racket.action_ball_policy_contract_sha256="
        + spec["policy_contract_sha256"]
    ) in argv
    assert not any(
        item
        == "task.racket.action_ball_policy_contract_sha256="
        + spec[L.SIGMA_VARIANT_IDENTITY_FIELD]
        for item in argv
    )
    for item in L.MONOTONIC_FRESH_CANARY_OVERRIDES:
        assert argv.count(item) == 1
    assert argv.count(
        L.SIGMA_PROFILE_ARG_PREFIX
        + L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE
    ) == 1
    assert not any(
        "sigma_update_every" in item
        or "sigma_ema_scale" in item
        or "sigma_pos_min" in item
        or "sigma_pos_max" in item
        or "sigma_vel_min" in item
        or "sigma_vel_max" in item
        or "sigma_normal_min" in item
        or "sigma_normal_max" in item
        for item in argv
    )
    assert not any(
        fragment in item
        for item in argv
        for fragment in (
            "checkpoint_path=",
            "checkpoint_tolerant=",
            "resume=",
            "load_run=",
            "load_checkpoint=",
        )
    )


def test_monotonic_sigma_canary_fails_closed_until_reward_recipe_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        None,
    )
    document = _spec(
        tmp_path,
        seed=0,
        stage="smoke",
        sigma_profile=L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE,
    )
    with pytest.raises(L.LaunchRefused, match="awaiting its code-pinned"):
        L._validate_spec_document(document)

    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        "e" * 64,
    )
    with pytest.raises(L.LaunchRefused, match="effective reward recipe differs"):
        L._validate_spec_document(document)


def test_monotonic_sigma_canary_rejects_bad_profile_namespace_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary_reward_sha = "e" * 64
    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        canary_reward_sha,
    )
    unknown = _spec(tmp_path, seed=0, stage="smoke")
    unknown[L.SIGMA_PROFILE_FIELD] = "operator_sigma"
    with pytest.raises(L.LaunchRefused, match="sigma_profile must be"):
        L._validate_spec_document(unknown)

    bad_namespace = _spec(
        tmp_path,
        seed=0,
        stage="smoke",
        sigma_profile=L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE,
    )
    bad_namespace["expected_effective_reward_recipe_sha256"] = canary_reward_sha
    bad_namespace["namespace"] = str(Path(bad_namespace["namespace"]).parent / "unnamed")
    bad_namespace["log_path"] = str(Path(bad_namespace["namespace"]) / "run.log")
    with pytest.raises(L.LaunchRefused, match="namespace must contain"):
        L._validate_spec_document(bad_namespace)

    forged_policy = _spec(
        tmp_path,
        seed=0,
        stage="smoke",
        sigma_profile=L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE,
    )
    forged_policy["expected_effective_reward_recipe_sha256"] = canary_reward_sha
    forged_policy[L.SIGMA_VARIANT_IDENTITY_FIELD] = "f" * 64
    with pytest.raises(L.LaunchRefused, match="scientific identity"):
        L._validate_spec_document(forged_policy)

    canary = _spec(
        tmp_path,
        seed=0,
        stage="smoke",
        sigma_profile=L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE,
    )
    canary["expected_effective_reward_recipe_sha256"] = canary_reward_sha
    spec = L._validate_spec_document(canary)
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited.append("+checkpoint_path=/tmp/forbidden.pt")
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: list(inherited)
    )
    with pytest.raises(L.LaunchRefused, match="fresh vendor log_std"):
        L._build_training_argv(spec, _bundle())


def test_monotonic_sigma_canary_refuses_even_materialized_block_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary_reward_sha = "e" * 64
    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        canary_reward_sha,
    )
    block = _materialized_block_config()
    original = L._action_config
    monkeypatch.setattr(
        L,
        "_action_config",
        lambda action_id: block if action_id == "bh_block" else original(action_id),
    )
    document = _spec(
        tmp_path,
        seed=0,
        stage="smoke",
        sigma_profile=L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE,
    )
    _select_block_lane(document)
    document["bundle"] = dict(
        L._R.require_materialized_pin(
            block.contact_bundle,
            action_id=block.action_id,
            layer="contact bundle",
        )
    )
    document[L.VENDOR_CONTRACT_FIELD] = block.runtime_contract.sha256
    document["expected_effective_reward_recipe_sha256"] = canary_reward_sha
    with pytest.raises(L.LaunchRefused, match="lane sigma_profile differs"):
        L._validate_spec_document(document)


def test_integrated_probe_argv_carries_stage_and_exact_budget(
    tmp_path: Path,
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="probe"))

    argv = L._build_training_argv(spec, _bundle())

    assert argv.count(L.STABLE_READY_PLANT_OVERRIDE) == 1
    assert argv.count(L.VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX + "probe") == 1
    assert "+n1_vendor_diagnostic_stage=push_evidence" not in argv
    assert argv.count(L.TABLE_ATTRIBUTION_PROBE_OVERRIDE) == 1
    assert argv.count(L.VENDOR_CONTRACT_ARG_PREFIX + VENDOR_CONTRACT_SHA) == 1
    assert "num_envs=4096" in argv
    assert "max_iterations=5" in argv
    assert "algo.runner.save_interval=1" in argv


def test_argv_adds_stable_ready_even_if_base_stops_supplying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited = [
        item
        for item in inherited
        if "task.domain_rand.stable_ready_plant" not in item
    ]
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: inherited
    )

    argv = L._build_training_argv(spec, _bundle())

    assert argv.count(L.STABLE_READY_PLANT_OVERRIDE) == 1


def test_argv_refuses_conflicting_stable_ready_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))
    inherited = L._base_build_training_argv(spec, _bundle())
    inherited = [
        "+task.domain_rand.stable_ready_plant=false"
        if item == L.STABLE_READY_PLANT_OVERRIDE
        else item
        for item in inherited
    ]
    monkeypatch.setattr(
        L, "_base_build_training_argv", lambda _spec, _bundle: inherited
    )

    with pytest.raises(L.LaunchRefused, match="conflicts with stable-ready"):
        L._build_training_argv(spec, _bundle())


def test_spec_cannot_inject_stable_ready_override(tmp_path: Path) -> None:
    spec = _spec(tmp_path, seed=0, stage="probe")
    spec["hydra_overrides"] = [
        "+task.domain_rand.stable_ready_plant=false"
    ]

    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_spec_document(spec)


def test_legacy_dynamic_ready_without_vendor_contract_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            _loop_dynamic_artifact(contract_sha=None),
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda _checkout, _commit, pin, **_kwargs: (
            dict(pin),
            tmp_path / pin["path"],
        ),
    )
    with pytest.raises(L.LaunchRefused, match="legacy bundle refused"):
        L._validate_vendor_runtime_binding(
            tmp_path, "a" * 40, _bundle(), VENDOR_CONTRACT_SHA
        )


def test_dynamic_ready_vendor_contract_must_match_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _loop_dynamic_artifact(contract_sha=VENDOR_CONTRACT_SHA)
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            artifact,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda _checkout, _commit, pin, **_kwargs: (
            dict(pin),
            tmp_path / pin["path"],
        ),
    )
    binding = L._validate_vendor_runtime_binding(
        tmp_path, "a" * 40, _bundle(), VENDOR_CONTRACT_SHA
    )
    assert binding["runtime_training_contract_sha256"] == VENDOR_CONTRACT_SHA

    with pytest.raises(
        L.LaunchRefused, match="action/motion differs|differs from spec"
    ):
        L._validate_vendor_runtime_binding(
            tmp_path, "a" * 40, _bundle(), "8" * 64
        )


@pytest.mark.parametrize(
    ("candidate_path", "accepted"),
    (
        (
            L._LOOP_ACTION_CONFIG.stable_motion.path,
            True,
        ),
        (
            "/workspace/old_checkout/"
            + L._LOOP_ACTION_CONFIG.stable_motion.path,
            True,
        ),
        (
            "prefix/" + L._LOOP_ACTION_CONFIG.stable_motion.path,
            False,
        ),
        (
            "/workspace/old_checkout/wrong_dir/"
            + Path(L._LOOP_ACTION_CONFIG.stable_motion.path).name,
            False,
        ),
        (
            "/workspace/old_checkout/../old_checkout/"
            + L._LOOP_ACTION_CONFIG.stable_motion.path,
            False,
        ),
        (
            "/workspace/old_checkout/./"
            + L._LOOP_ACTION_CONFIG.stable_motion.path,
            False,
        ),
        (
            "/workspace/old_checkout//"
            + L._LOOP_ACTION_CONFIG.stable_motion.path,
            False,
        ),
        (
            "/workspace/old_checkout/"
            + L._LOOP_ACTION_CONFIG.stable_motion.path
            + "/",
            False,
        ),
        (
            "/workspace/old\tcheckout/"
            + L._LOOP_ACTION_CONFIG.stable_motion.path,
            False,
        ),
        (
            "//workspace/old_checkout/"
            + L._LOOP_ACTION_CONFIG.stable_motion.path,
            False,
        ),
    ),
)
def test_dynamic_ready_motion_provenance_uses_full_logical_repo_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_path: str,
    accepted: bool,
) -> None:
    artifact = _loop_dynamic_artifact(contract_sha=VENDOR_CONTRACT_SHA)
    artifact["sources"]["stable_motion"]["path"] = candidate_path
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            artifact,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda _checkout, _commit, pin, **_kwargs: (
            dict(pin),
            tmp_path / pin["path"],
        ),
    )
    if accepted:
        result = L._validate_vendor_runtime_binding(
            tmp_path,
            "a" * 40,
            _bundle(),
            VENDOR_CONTRACT_SHA,
        )
        assert result["stable_motion_path"] == (
            L._LOOP_ACTION_CONFIG.stable_motion.path
        )
        return
    with pytest.raises(L.LaunchRefused, match="action/motion differs"):
        L._validate_vendor_runtime_binding(
            tmp_path,
            "a" * 40,
            _bundle(),
            VENDOR_CONTRACT_SHA,
        )


def test_dynamic_ready_wrong_motion_sha_is_refused_before_tracked_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _loop_dynamic_artifact(contract_sha=VENDOR_CONTRACT_SHA)
    artifact["sources"]["stable_motion"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            artifact,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: pytest.fail("tracked read must not run"),
    )
    with pytest.raises(L.LaunchRefused, match="action/motion differs"):
        L._validate_vendor_runtime_binding(
            tmp_path,
            "a" * 40,
            _bundle(),
            VENDOR_CONTRACT_SHA,
        )


def test_dynamic_ready_revalidates_exact_registry_motion_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _loop_dynamic_artifact(contract_sha=VENDOR_CONTRACT_SHA)
    artifact["sources"]["stable_motion"]["path"] = (
        "/workspace/old_checkout/"
        + L._LOOP_ACTION_CONFIG.stable_motion.path
    )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            artifact,
        ),
    )
    observed = {}

    def _reject_drift(_checkout, _commit, pin, **_kwargs):
        observed.update(pin)
        raise L.LaunchRefused("stable motion SHA differs")

    monkeypatch.setattr(L._B, "_verify_tracked_file", _reject_drift)
    with pytest.raises(L.LaunchRefused, match="stable motion SHA differs"):
        L._validate_vendor_runtime_binding(
            tmp_path,
            "a" * 40,
            _bundle(),
            VENDOR_CONTRACT_SHA,
        )
    assert observed == dict(
        L._R.stable_pin(L._LOOP_ACTION_CONFIG.stable_motion)
    )


@pytest.mark.parametrize(
    ("selected_action", "candidate_action"),
    (("bh_loop_c", "bh_block"), ("bh_block", "bh_loop_c")),
)
def test_dynamic_ready_cross_action_substitution_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_action: str,
    candidate_action: str,
) -> None:
    candidate_config = L._R.get_action_config(candidate_action)
    artifact = {
        "action_id": candidate_action,
        "sources": {
            "stable_motion": dict(
                L._R.stable_pin(candidate_config.stable_motion)
            ),
            "runtime_training_contract": {"sha256": "5" * 64},
        },
    }
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            artifact,
        ),
    )
    with pytest.raises(L.LaunchRefused, match="action/motion differs"):
        L._validate_vendor_runtime_binding(
            tmp_path,
            "a" * 40,
            _bundle(),
            "5" * 64,
            action_id=selected_action,
        )


def test_required_identity_cross_action_substitution_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    block = _materialized_block_config()
    original = L._action_config
    monkeypatch.setattr(
        L,
        "_action_config",
        lambda action_id: block if action_id == "bh_block" else original(action_id),
    )
    loop_manifest = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / L.VENDOR_IDENTITY_MANIFEST_SOURCE
        ).read_text(encoding="utf-8")
    )
    loop_manifest["robot_action_contract"]["groups"] = _explicit_identity_groups(
        Path(__file__).resolve().parents[3]
    )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                "path": block.required_identity_manifest.path,
                "sha256": block.required_identity_manifest.sha256,
            },
            loop_manifest,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (args[2], tmp_path / args[2]["path"]),
    )
    with pytest.raises(L.LaunchRefused, match="awaiting exact runtime"):
        L._validate_vendor_identity_manifest(
            Path(__file__).resolve().parents[3],
            "a" * 40,
            action_id="bh_block",
        )


def test_vendor_bundle_gate_rejects_v1_and_accepts_v2_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    motion_sha = "e" * 64
    candidate = {
        "schema_version": 2,
        "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
        "action_id": "bh_loop_c",
        "robot": {"family": "AgiBot A3"},
        "sources": {
            "stable_motion": {
                "frame_index": 0,
                "sha256": motion_sha,
            }
        },
        "runtime_plant": {},
        "required_next_gate": {
            "kind": L._B.NOMINAL_HOLD_RECEIPT_KIND,
            "zero_terminal_required": ["joint_actual_forbidden"],
        },
    }
    candidate["content_sha256"] = L._B._canonical_ascii_sha256(candidate)
    artifact_pin = {"path": "ready.v2.json", "sha256": "f" * 64}
    receipt = {
        "schema_version": 1,
        "kind": L._B.NOMINAL_HOLD_RECEIPT_KIND,
        "verdict": "PASS",
        "action_id": "bh_loop_c",
        "motion_sha256": motion_sha,
        "plant_contract_match": True,
        "terminal_reasons": [],
        "generic_terminated": False,
        "generic_truncated": False,
        "artifact": {
            "sha256": artifact_pin["sha256"],
            "content_sha256": candidate["content_sha256"],
        },
        "active_terminations": ["joint_actual_forbidden"],
    }
    receipt["content_sha256"] = L._B.canonical_sha256(receipt)
    values = iter(
        (
            (artifact_pin, candidate),
            ({"path": "hold.json", "sha256": "0" * 64}, receipt),
        )
    )
    monkeypatch.setattr(
        L._B, "_load_tracked_json", lambda *args, **kwargs: next(values)
    )
    result = L._validate_vendor_dynamic_ready(
        tmp_path,
        "a" * 40,
        {
            "artifact": artifact_pin,
            "nominal_hold_receipt": {
                "path": "hold.json",
                "sha256": "0" * 64,
            },
        },
        action_id="bh_loop_c",
        motion_sha256=motion_sha,
    )
    assert result["artifact"] == artifact_pin

    legacy = dict(candidate)
    legacy["schema_version"] = 1
    legacy["kind"] = "agibot_a3_action_dynamic_ready_candidate_v1"
    legacy["content_sha256"] = L._B._canonical_ascii_sha256(
        {key: value for key, value in legacy.items() if key != "content_sha256"}
    )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (artifact_pin, legacy),
    )
    with pytest.raises(L.LaunchRefused, match="schema-v2"):
        L._validate_vendor_dynamic_ready(
            tmp_path,
            "a" * 40,
            {
                "artifact": artifact_pin,
                "nominal_hold_receipt": {
                    "path": "hold.json",
                    "sha256": "0" * 64,
                },
            },
            action_id="bh_loop_c",
            motion_sha256=motion_sha,
        )


def test_repo_real_old_bh_loop_dynamic_ready_is_rejected() -> None:
    checkout = Path(__file__).resolve().parents[3]
    relative = "configs/a3_dynamic_ready_20260730/bh_loop_c.dynamic_ready.v1.json"
    artifact = checkout / relative
    pin = {
        "path": relative,
        "sha256": L._B.sha256_file(artifact),
    }
    bundle = _bundle()
    bundle["dynamic_ready"]["artifact"] = pin
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    with pytest.raises(
        L.LaunchRefused, match="action/motion differs|differs from spec"
    ):
        L._validate_vendor_runtime_binding(
            checkout, commit, bundle, VENDOR_CONTRACT_SHA
        )


def test_tracked_identity_resolves_materialized_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    manifest_path = checkout / L.VENDOR_IDENTITY_MANIFEST_SOURCE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert L._B.sha256_file(manifest_path) == L.VENDOR_IDENTITY_MANIFEST_SHA256
    manifest["robot_action_contract"]["groups"] = _explicit_identity_groups(
        checkout
    )
    # Source closure is revalidated by the launcher against the selected clean
    # commit.  This host unit isolates action-specific identity/contract
    # selection because any source edit intentionally requires rematerializing
    # the tracked identity manifest.
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            manifest,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (args[2], checkout / args[2]["path"]),
    )

    validated = L._validate_vendor_identity_manifest(checkout, "a" * 40)
    assert validated == {
        "manifest": {
            "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
            "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
        },
        "runtime_training_contract_sha256": manifest[
            "runtime_materialization"
        ]["training_contract_sha256"],
    }


def test_tracked_identity_still_refuses_awaiting_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    manifest_path = checkout / L.VENDOR_IDENTITY_MANIFEST_SOURCE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["robot_action_contract"]["groups"] = _explicit_identity_groups(
        checkout
    )
    manifest["status"] = "awaiting_runtime_materialization"
    manifest["runtime_materialization"]["training_contract_sha256"] = None
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            manifest,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (args[2], checkout / args[2]["path"]),
    )

    with pytest.raises(L.LaunchRefused, match="awaiting exact runtime"):
        L._validate_vendor_identity_manifest(checkout, "a" * 40)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "exactly 31 unique"),
        ("duplicate", "exactly 31 unique"),
        ("nonfinite", "finite number"),
        ("regex", "explicit joint names"),
        ("group_count", "exactly 12 groups"),
    ),
)
def test_required_identity_explicit_group_negative_cases(
    monkeypatch: pytest.MonkeyPatch, mutation: str, message: str
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    manifest_path = checkout / L.VENDOR_IDENTITY_MANIFEST_SOURCE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = _explicit_identity_groups(checkout)
    if mutation == "missing":
        groups[-1]["joints"].pop()
    elif mutation == "duplicate":
        groups[-1]["joints"][-1] = groups[0]["joints"][0]
    elif mutation == "nonfinite":
        groups[0]["armature"] = float("inf")
    elif mutation == "regex":
        groups[0]["joints"][0] = ".*_hip_pitch_joint"
    else:
        split_joint = groups[0]["joints"].pop()
        split_group = dict(groups[0])
        split_group["joints"] = [split_joint]
        groups.append(split_group)
    manifest["robot_action_contract"]["groups"] = groups
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            manifest,
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (args[2], checkout / args[2]["path"]),
    )
    with pytest.raises(L.LaunchRefused, match=message):
        L._validate_vendor_identity_manifest(checkout, "a" * 40)


def test_vendor_wrapper_replaces_legacy_robot_source_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(L._B, "sha256_file", lambda path: "7" * 64)
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda root, commit, pin, **kwargs: (pin, root / pin["path"]),
    )

    sources = L._validate_runtime_sources(checkout, "a" * 40)
    assert sources["vendor A3 robot source"]["path"] == L.ROBOT_SOURCE
    assert "historical N1 stable-ready robot source" not in sources


def _runtime_template_argv(
    tmp_path: Path,
    *,
    output: Path,
    lane: str = L.LOOP_STATIC_LANE,
    stage: str = "smoke",
) -> list[str]:
    checkout = tmp_path / "template-checkout"
    checkout.mkdir(exist_ok=True)
    python = tmp_path / "isaac-python"
    if not python.exists():
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o755)
    runs = tmp_path / "template-runs"
    runs.mkdir(exist_ok=True)
    namespace = runs / f"run-{lane}-{stage}"
    return [
        "template",
        "--lane",
        lane,
        "--stage",
        stage,
        "--output",
        str(output),
        "--checkout",
        str(checkout),
        "--commit-sha",
        "a" * 40,
        "--isaac-python",
        str(python),
        "--gpu-index",
        "0",
        "--gpu-uuid",
        "GPU-template-0",
        "--owner",
        "Franco",
        "--namespace",
        str(namespace),
    ]


def test_template_parser_exposes_only_lane_and_operational_axes() -> None:
    parser = L._parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, L.argparse._SubParsersAction)
    )
    template = subparsers.choices["template"]
    flags = {
        flag
        for action in template._actions
        for flag in action.option_strings
    }
    for forbidden in (
        "--action-id",
        "--sigma-profile",
        "--policy-contract-sha256",
        "--effective-reward-recipe-sha256",
        "--seed",
        "--num-envs",
        "--max-iterations",
        "--save-interval",
    ):
        assert forbidden not in flags
    assert "--lane" in flags


def test_template_fails_closed_while_policy_pin_is_unmaterialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(L, "BH_LOOP_C_BASE_POLICY_CONTRACT_SHA256", None)
    args = L._parser().parse_args(
        _runtime_template_argv(tmp_path, output=tmp_path / "spec.json")
    )
    with pytest.raises(L.LaunchRefused, match="policy contract materialization"):
        L.materialize_template(args)


@pytest.mark.parametrize(
    "lane_id",
    (L.LOOP_STATIC_LANE, L.LOOP_ADAPTIVE_LANE),
)
def test_template_fails_closed_while_reward_pin_is_unmaterialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lane_id: str
) -> None:
    if lane_id == L.LOOP_STATIC_LANE:
        monkeypatch.setattr(L, "STATIC_EFFECTIVE_REWARD_RECIPE_SHA256", None)
    else:
        monkeypatch.setattr(
            L,
            "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
            None,
        )
    args = L._parser().parse_args(
        _runtime_template_argv(
            tmp_path, output=tmp_path / "spec.json", lane=lane_id
        )
    )
    with pytest.raises(
        L.LaunchRefused, match="effective reward recipe materialization"
    ):
        L.materialize_template(args)


@pytest.mark.parametrize(
    "registry_field",
    ("fixed_domain_initial_receipt", "reward_economy_receipt"),
)
def test_template_fails_closed_while_required_receipt_pin_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry_field: str
) -> None:
    action = replace(
        _TEST_LOOP_CONFIG,
        **{
            registry_field: L._R.ArtifactPin(
                getattr(_TEST_LOOP_CONFIG, registry_field).path, None
            )
        },
    )
    monkeypatch.setattr(
        L,
        "_action_config",
        lambda action_id: action
        if action_id == action.action_id
        else L._R.ACTION_CONFIGS[action_id],
    )

    with pytest.raises(L.LaunchRefused, match="awaiting code-pinned"):
        L._lane_scientific_spec(L.LOOP_STATIC_LANE, "smoke")
    with pytest.raises(L.LaunchRefused, match="awaiting code-pinned"):
        L._validate_spec_document(_spec(tmp_path, seed=0, stage="smoke"))


def test_prelaunch_receipts_are_consumed_and_bound_to_claim_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed, economy = _prelaunch_receipt_documents()
    _install_prelaunch_receipt_fakes(monkeypatch, fixed, economy)
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="probe"))

    observed = L._validate_prelaunch_receipts(
        Path(spec["source"]["checkout"]),
        spec["source"]["commit_sha"],
        spec,
    )

    assert observed["fixed_domain"]["action_id"] == "bh_loop_c"
    assert observed["fixed_domain"]["lane_id"] == L.LOOP_STATIC_LANE
    assert observed["fixed_domain"]["domain_epoch"] == 0
    assert observed["reward_ppo_economy"]["reward_global_scalar"] == 1.0
    assert observed["reward_ppo_economy"]["noise_std_type"] == "log"
    assert observed["reward_ppo_economy"]["required_initial_sigma"] == 0.02


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("domain_epoch", "exact action/lane/r6 source identity"),
        ("identity_path", "exact action/lane/r6 source identity"),
        ("scalar_std", "exact r6 scale/log_std sources"),
        ("wrong_reward_sha", "exact r6 scale/log_std sources"),
        ("training_authority", "exact r6 scale/log_std sources"),
    ),
)
def test_prelaunch_receipt_semantic_drift_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixed, economy = _prelaunch_receipt_documents()
    if mutation == "domain_epoch":
        fixed["content"]["domain_epoch"] = 1
    elif mutation == "identity_path":
        fixed["content"]["registry_action_source_identity"] = copy.deepcopy(
            fixed["content"]["registry_action_source_identity"]
        )
        fixed["content"]["registry_action_source_identity"]["planned_paths"][
            "contact_bundle"
        ] = "configs/wrong_epoch/bundle.json"
    elif mutation == "scalar_std":
        economy["ppo_economy"]["required_final_policy"][
            "noise_std_type"
        ] = "scalar"
    elif mutation == "wrong_reward_sha":
        economy["reward_economy"]["effective_reward_recipe_sha256"] = "0" * 64
    else:
        economy["authorization"]["training"] = True
    _install_prelaunch_receipt_fakes(monkeypatch, fixed, economy)
    spec = L._validate_spec_document(_spec(tmp_path, seed=0, stage="probe"))

    with pytest.raises(L.LaunchRefused, match=message):
        L._validate_prelaunch_receipts(
            Path(spec["source"]["checkout"]),
            spec["source"]["commit_sha"],
            spec,
        )


def test_three_lane_table_owns_action_sigma_policy_reward_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _materialized_block_config()
    original = L._action_config
    monkeypatch.setattr(
        L,
        "_action_config",
        lambda action_id: block if action_id == "bh_block" else original(action_id),
    )
    canary_reward = "e" * 64
    monkeypatch.setattr(
        L,
        "MONOTONIC_FRESH_CANARY_EFFECTIVE_REWARD_RECIPE_SHA256",
        canary_reward,
    )
    loop = L._lane_scientific_spec(L.LOOP_STATIC_LANE, "probe")
    block_spec = L._lane_scientific_spec(L.BLOCK_STATIC_LANE, "probe")
    adaptive = L._lane_scientific_spec(L.LOOP_ADAPTIVE_LANE, "probe")
    assert (loop["action_id"], block_spec["action_id"], adaptive["action_id"]) == (
        "bh_loop_c",
        "bh_block",
        "bh_loop_c",
    )
    assert loop[L.SIGMA_PROFILE_FIELD] == block_spec[L.SIGMA_PROFILE_FIELD] == (
        L.STATIC_SIGMA_PROFILE
    )
    assert adaptive[L.SIGMA_PROFILE_FIELD] == (
        L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE
    )
    assert adaptive["policy_contract_sha256"] == loop["policy_contract_sha256"]
    assert adaptive["expected_effective_reward_recipe_sha256"] == canary_reward
    assert {loop["seed"], block_spec["seed"], adaptive["seed"]} == {0}


def test_runtime_template_is_canonical_repeatable_and_no_clobber(
    tmp_path: Path,
) -> None:
    output_a = tmp_path / "a.json"
    output_b = tmp_path / "b.json"
    args_a = L._parser().parse_args(
        _runtime_template_argv(tmp_path, output=output_a)
    )
    args_b = L._parser().parse_args(
        _runtime_template_argv(tmp_path, output=output_b)
    )
    result_a = L.materialize_template(args_a)
    result_b = L.materialize_template(args_b)
    assert output_a.read_bytes() == output_b.read_bytes()
    assert result_a["sha256"] == result_b["sha256"]
    document = json.loads(output_a.read_text(encoding="utf-8"))
    assert output_a.read_bytes() == _canonical(document)
    assert L._validate_spec_document(document) == document
    assert document[L.VENDOR_LANE_FIELD] == L.LOOP_STATIC_LANE
    assert document[L.SIGMA_PROFILE_FIELD] == L.STATIC_SIGMA_PROFILE
    assert document[L.SIGMA_VARIANT_IDENTITY_FIELD] == "c" * 64
    assert document["seed"] == 0
    assert (
        document["num_envs"],
        document["max_iterations"],
        document["save_interval"],
    ) == L.EXACT_STAGE_BUDGETS["smoke"]
    with pytest.raises(L.LaunchRefused, match="no-clobber"):
        L.materialize_template(args_a)


def test_template_rejects_symlink_output_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    args = L._parser().parse_args(
        _runtime_template_argv(tmp_path, output=alias / "spec.json")
    )
    with pytest.raises(L.LaunchRefused, match="output parent"):
        L.materialize_template(args)


def test_runtime_template_rejects_non_uuid_gpu_before_write(tmp_path: Path) -> None:
    output = tmp_path / "bad-gpu.json"
    argv = _runtime_template_argv(tmp_path, output=output)
    argv[argv.index("--gpu-uuid") + 1] = "not-a-gpu"
    args = L._parser().parse_args(argv)
    with pytest.raises(L.LaunchRefused, match="explicit GPU UUID"):
        L.materialize_template(args)
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("action_id", "bh_block", "lane action_id"),
        (L.SIGMA_PROFILE_FIELD, L.MONOTONIC_FRESH_CANARY_SIGMA_PROFILE, "lane sigma"),
        ("policy_contract_sha256", "9" * 64, "policy contract differs"),
        (
            "expected_effective_reward_recipe_sha256",
            "9" * 64,
            "effective reward recipe differs",
        ),
        ("seed", 1, "lane seed"),
        ("max_iterations", 3, "exactly 1 env / 2 updates"),
    ),
)
def test_lane_validator_rejects_scientific_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    document = _spec(tmp_path, seed=0, stage="smoke")
    document[field] = value
    with pytest.raises(L.LaunchRefused, match=message):
        L._validate_spec_document(document)


def _integrated_probe_acceptance() -> dict[str, bool]:
    return {
        "integrated_probe_exact_pass": True,
        "finite_checkpoints": True,
        "normalizer_checkpoint_persistence": True,
        "runtime_abi_exact": True,
        "fresh_log_std_initialization_exact": True,
        "control_step_delay_exact": True,
        "positive_policy_std_and_finite_lr": True,
        "reward_ppo_economy_runtime_pass": True,
        "zero_actual_hard_edge": True,
        "zero_qdes_edge": True,
        "zero_nonfinite": True,
        "push_timer_lower_bound_crossed": True,
        "velocity_only_push_observed_nonzero": True,
        "velocity_push_six_axis_extrema_finite_and_in_range": True,
        "natural_training_completion": True,
    }


def _gate_receipt_for_template(
    *,
    lane: str,
    receipt_relative: str,
    skeleton_relative: str,
) -> dict:
    scientific = L._lane_scientific_spec(
        lane,
        "long",
        gate_pin={"path": receipt_relative, "sha256": "0" * 64},
    )
    receipt = {
        "schema_version": 3,
        "kind": L.VENDOR_PROBE_GATE_KIND,
        "verdict": "PASS",
        "producer": {"algorithm": "exact_integrated_probe_v3"},
        "evidence_source_commit": "a" * 40,
        "scientific_identity": {
            "action_id": scientific["action_id"],
            "scope": scientific["scope"],
            "seed": scientific["seed"],
            "policy_contract_sha256": scientific["policy_contract_sha256"],
            "sigma_profile": scientific[L.SIGMA_PROFILE_FIELD],
            "sigma_variant_scientific_identity_sha256": scientific[
                L.SIGMA_VARIANT_IDENTITY_FIELD
            ],
            "effective_reward_recipe_sha256": scientific[
                "expected_effective_reward_recipe_sha256"
            ],
            L.VENDOR_CONTRACT_FIELD: scientific[L.VENDOR_CONTRACT_FIELD],
            L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD: scientific[
                L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD
            ],
            L.REWARD_ECONOMY_RECEIPT_FIELD: scientific[
                L.REWARD_ECONOMY_RECEIPT_FIELD
            ],
        },
        "stages": {"probe": {}},
        "acceptance": _integrated_probe_acceptance(),
        "successor_policy": {
            "required_gate_source_ancestor_commit": "a" * 40,
            "allowed_artifact_descendant_diff": {
                "exact_paths": [receipt_relative, skeleton_relative],
                "prefixes": ["docs/"],
            },
        },
        "authorization": {},
    }
    receipt["content_sha256"] = L.canonical_sha256(receipt)
    return receipt


def test_long_consumer_accepts_only_integrated_probe_receipt_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific = L._lane_scientific_spec(L.LOOP_STATIC_LANE, "probe")
    identity = {
        "action_id": scientific["action_id"],
        "scope": scientific["scope"],
        "seed": scientific["seed"],
        "policy_contract_sha256": scientific["policy_contract_sha256"],
        L.SIGMA_PROFILE_FIELD: scientific[L.SIGMA_PROFILE_FIELD],
        L.SIGMA_VARIANT_IDENTITY_FIELD: scientific[
            L.SIGMA_VARIANT_IDENTITY_FIELD
        ],
        "effective_reward_recipe_sha256": scientific[
            "expected_effective_reward_recipe_sha256"
        ],
        L.VENDOR_CONTRACT_FIELD: scientific[L.VENDOR_CONTRACT_FIELD],
        L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD: scientific[
            L.FIXED_DOMAIN_INITIAL_RECEIPT_FIELD
        ],
        L.REWARD_ECONOMY_RECEIPT_FIELD: scientific[
            L.REWARD_ECONOMY_RECEIPT_FIELD
        ],
        "scientific_argv_canonical_sha256": "9" * 64,
    }
    stage = {
        "stage": "probe",
        "namespace": str(tmp_path / "probe"),
        "run_directory": str(tmp_path / "run"),
    }
    producer_pin = {
        "path": L.VENDOR_PROBE_GATE_PRODUCER_SOURCE,
        "sha256": "2" * 64,
    }
    receipt_pin = {"path": "configs/gate.json", "sha256": "1" * 64}
    receipt = {
        "schema_version": 3,
        "kind": L.VENDOR_PROBE_GATE_KIND,
        "verdict": "PASS",
        "producer": {
            "source": producer_pin,
            "gate_source_commit": "a" * 40,
            "algorithm": "exact_integrated_probe_v3",
            "self_reference_free": True,
        },
        "evidence_source_commit": "a" * 40,
        "scientific_identity": identity,
        "stages": {"probe": stage},
        "acceptance": _integrated_probe_acceptance(),
        "successor_policy": {},
        "authorization": {
            "vendor_n1_long_launch": True,
            "formal_evidence": False,
            "curriculum_promotion": False,
            "resume": False,
            "export": False,
            "judge": False,
            "deployment": False,
            "hardware": False,
        },
        "content_sha256": "3" * 64,
    }

    class GateModule:
        class ReceiptRefused(RuntimeError):
            pass

        @staticmethod
        def _scientific_identity(_payload):
            return identity

        @staticmethod
        def _stage_evidence(_namespace, _run_directory, *, expected_stage):
            assert expected_stage == "probe"
            return stage, identity

    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (receipt_pin, receipt),
    )
    monkeypatch.setattr(L._B, "_verify_content_seal", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda *args, **kwargs: (producer_pin, tmp_path / "producer.py"),
    )
    monkeypatch.setattr(L, "_load_probe_gate_module", lambda _checkout: GateModule)
    monkeypatch.setattr(L, "_validate_probe_gate_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        L, "_validate_probe_gate_descendant_policy", lambda *args, **kwargs: None
    )

    observed = L._validate_vendor_probe_gate_receipt(
        tmp_path,
        "b" * 40,
        receipt_pin,
        spec={},
        payload={},
    )
    assert observed["pin"] == receipt_pin

    receipt["stages"]["push_evidence"] = dict(stage, stage="push_evidence")
    with pytest.raises(L.LaunchRefused, match="keys differ"):
        L._validate_vendor_probe_gate_receipt(
            tmp_path,
            "b" * 40,
            receipt_pin,
            spec={},
            payload={},
        )


def test_long_scientific_skeleton_and_runtime_merge_avoid_git_fixed_point(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "artifact-checkout"
    receipt_relative = (
        "configs/n1_vendor_probe_gate_20260731/loop.pass.json"
    )
    skeleton_relative = (
        "configs/n1_vendor_launch_20260731/"
        "bh_loop_c_static_v1.long.seed0.json"
    )
    receipt_path = checkout / receipt_relative
    skeleton_path = checkout / skeleton_relative
    receipt_path.parent.mkdir(parents=True)
    skeleton_path.parent.mkdir(parents=True)
    receipt = _gate_receipt_for_template(
        lane=L.LOOP_STATIC_LANE,
        receipt_relative=receipt_relative,
        skeleton_relative=skeleton_relative,
    )
    receipt_path.write_bytes(_canonical(receipt))

    scientific_args = L._parser().parse_args(
        [
            "template",
            "--lane",
            L.LOOP_STATIC_LANE,
            "--stage",
            "long",
            "--scientific-only",
            "--checkout",
            str(checkout),
            "--probe-gate-receipt",
            str(receipt_path),
            "--output",
            str(skeleton_path),
        ]
    )
    L.materialize_template(scientific_args)
    skeleton = json.loads(skeleton_path.read_text(encoding="utf-8"))
    assert not (set(skeleton) & L._OPERATIONAL_SPEC_KEYS)
    assert skeleton[L.VENDOR_PROBE_GATE_FIELD]["path"] == receipt_relative

    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-qm", "artifact"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python = tmp_path / "long-python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    runs = tmp_path / "long-runs"
    runs.mkdir()
    namespace = runs / f"run-{L.LOOP_STATIC_LANE}-long"
    runtime_output = tmp_path / "control" / "long.json"
    runtime_output.parent.mkdir()
    runtime_args = L._parser().parse_args(
        [
            "template",
            "--lane",
            L.LOOP_STATIC_LANE,
            "--stage",
            "long",
            "--scientific-template",
            str(skeleton_path),
            "--checkout",
            str(checkout),
            "--commit-sha",
            commit,
            "--isaac-python",
            str(python),
            "--gpu-index",
            "1",
            "--gpu-uuid",
            "GPU-long-1",
            "--owner",
            "Franco",
            "--namespace",
            str(namespace),
            "--output",
            str(runtime_output),
        ]
    )
    L.materialize_template(runtime_args)
    runtime = json.loads(runtime_output.read_text(encoding="utf-8"))
    assert runtime["source"]["commit_sha"] == commit
    assert L._scientific_projection(runtime) == skeleton
    assert runtime_output.relative_to(tmp_path / "control") == Path("long.json")


def test_scientific_skeleton_rejects_wrong_successor_and_identity(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    receipt_relative = (
        "configs/n1_vendor_probe_gate_20260731/loop.pass.json"
    )
    skeleton_relative = (
        "configs/n1_vendor_launch_20260731/loop.long.json"
    )
    receipt_path = checkout / receipt_relative
    output = checkout / skeleton_relative
    receipt_path.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    receipt = _gate_receipt_for_template(
        lane=L.LOOP_STATIC_LANE,
        receipt_relative=receipt_relative,
        skeleton_relative=skeleton_relative,
    )
    receipt["scientific_identity"]["seed"] = 1
    unsigned = dict(receipt)
    unsigned.pop("content_sha256")
    receipt["content_sha256"] = L.canonical_sha256(unsigned)
    receipt_path.write_bytes(_canonical(receipt))
    args = L._parser().parse_args(
        [
            "template",
            "--lane",
            L.LOOP_STATIC_LANE,
            "--stage",
            "long",
            "--scientific-only",
            "--checkout",
            str(checkout),
            "--probe-gate-receipt",
            str(receipt_path),
            "--output",
            str(output),
        ]
    )
    with pytest.raises(L.LaunchRefused, match="scientific identity differs"):
        L.materialize_template(args)


def test_integrated_probe_runtime_sources_are_exact_and_auditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_manager = tmp_path / "event_manager.py"
    push_events = tmp_path / "events.py"
    event_manager.write_bytes(b"interval semantics\n")
    push_events.write_bytes(b"velocity push semantics\n")
    paths = {
        "IsaacLab interval event manager": event_manager,
        "IsaacLab push-by-velocity event": push_events,
    }
    pins = {
        label: {"path": str(path), "sha256": L._B.sha256_file(path)}
        for label, path in paths.items()
    }
    monkeypatch.setattr(L, "INTEGRATED_PROBE_RUNTIME_SOURCE_PINS", pins)
    monkeypatch.setattr(
        L, "_integrated_probe_runtime_source_origins", lambda: paths
    )

    observed = L._validate_integrated_probe_runtime_sources()

    assert observed == pins


def test_integrated_probe_uses_one_to_three_second_finite_six_axis_velocity_push(
) -> None:
    checkout = Path(__file__).resolve().parents[3]
    gate = L._load_probe_gate_module(checkout)
    axes = {
        axis: {
            "below_range_count": 0,
            "above_range_count": 0,
            "observed_delta_min": -0.1,
            "observed_delta_max": -0.05,
        }
        for axis in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    records = [
        {
            "event": "hope_push_velocity_diagnostic_update",
            "schema_version": 1,
            "ppo_update": update,
            "counters": {
                "event_call_count": 1,
                "env_application_count": 1,
                "delta_nonfinite_element_count": 0,
                "axes": copy.deepcopy(axes),
            },
        }
        for update in range(5)
    ]

    observed = gate._validate_push_velocity(
        records,
        stage="probe",
        updates=5,
        num_envs=4096,
    )

    assert gate.PUSH_INTERVAL_RANGE_S == (1.0, 3.0)
    assert set(observed["axis_extrema"]) == set(axes)
    assert observed["aggregate"][
        "six_axis_extrema_finite_and_in_range"
    ] is True
    assert observed["aggregate"]["event_call_count"] == 5
    assert observed["aggregate"]["env_application_count"] == 5
    assert "six_axis_signed_variation" not in observed["aggregate"]


def test_integrated_probe_runtime_source_drift_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        "IsaacLab interval event manager": tmp_path / "event_manager.py",
        "IsaacLab push-by-velocity event": tmp_path / "events.py",
    }
    for path in paths.values():
        path.write_bytes(b"runtime source\n")
    pins = {
        label: {"path": str(path), "sha256": "0" * 64}
        for label, path in paths.items()
    }
    monkeypatch.setattr(L, "INTEGRATED_PROBE_RUNTIME_SOURCE_PINS", pins)
    monkeypatch.setattr(
        L, "_integrated_probe_runtime_source_origins", lambda: paths
    )

    with pytest.raises(L.LaunchRefused, match="runtime source SHA differs"):
        L._validate_integrated_probe_runtime_sources()


def test_launch_rechecks_integrated_probe_sources_before_base_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {"runtime": {"path": "/runtime.py", "sha256": "a" * 64}}
    receipts = {"fixed_domain": {"content_sha256": "b" * 64}}
    payload = {
        "spec": {
            "stage": "probe",
            "source": {"checkout": "/checkout", "commit_sha": "c" * 40},
        },
        L.INTEGRATED_PROBE_CLAIM_FIELD: sources,
        L.PRELAUNCH_RECEIPTS_CLAIM_FIELD: receipts,
    }
    calls: list[str] = []
    monkeypatch.setattr(
        L,
        "_validate_integrated_probe_runtime_sources",
        lambda: calls.append("sources") or sources,
    )
    monkeypatch.setattr(
        L,
        "_base_launch",
        lambda plan, confirm_claim: calls.append("base") or {},
    )
    monkeypatch.setattr(
        L,
        "_validate_prelaunch_receipts",
        lambda *args, **kwargs: calls.append("receipts") or receipts,
    )

    result = L.launch(
        {"canonical_payload": payload}, confirm_claim="a" * 64
    )

    assert calls == ["sources", "receipts", "base"]
    assert result["kind"] == "n1_vendor_baseline_diagnostic_launch_result_v1"


def test_non_probe_claim_cannot_carry_integrated_probe_runtime_sources() -> None:
    with pytest.raises(L.LaunchRefused, match="non-probe vendor claim"):
        L._revalidate_integrated_probe_claim_sources(
            {
                "spec": {"stage": "smoke"},
                L.INTEGRATED_PROBE_CLAIM_FIELD: {},
            }
        )


def test_actual_authority_receipt_pin_matches_materialized_file() -> None:
    checkout = Path(__file__).resolve().parents[3]
    receipt_path = checkout / _TEST_LOOP_CONFIG.runtime_authority_receipt.path
    assert L.VENDOR_AUTHORITY_RECEIPT_SHA256 is not None
    assert L.VENDOR_AUTHORITY_RECEIPT_SHA256 == L._B.sha256_file(receipt_path)


def test_actual_authority_loader_and_full_candidate_validator_are_both_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    class AuthorityError(RuntimeError):
        pass

    def load(receipt_path, **kwargs):
        calls.append(("load", kwargs))
        return {
            "receipt_path": "authority.json",
            "receipt_sha256": "a" * 64,
            "runtime_training_contract": {
                "path": "contract.json",
                "sha256": VENDOR_CONTRACT_SHA,
                "schema_version": 3,
            },
            "runtime_plant_identity": {"authority": True},
            "verified_vendor_runtime": {
                "action_id": "bh_loop_c",
                "motion_sha256": L._LOOP_ACTION_CONFIG.stable_motion.sha256,
            },
        }

    def validate(candidate, authority, **kwargs):
        calls.append(("validate", candidate))
        assert authority["runtime_plant_identity"] == {"authority": True}
        assert kwargs == {"action_id": "bh_loop_c"}
        return {"full_candidate_plant": True}

    module = SimpleNamespace(
        RECEIPT_REPO_PATH="configs/authority.json",
        VendorRuntimeAuthorityError=AuthorityError,
        load_and_validate_vendor_runtime_authority=load,
        validate_candidate_runtime_plant_against_vendor_authority=validate,
    )
    monkeypatch.setattr(L, "_load_vendor_authority_module", lambda root: module)
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.v2.json", "sha256": "b" * 64},
            {
                "schema_version": 2,
                "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
                "action_id": "bh_loop_c",
                "sources": {
                    "stable_motion": dict(
                        L._R.stable_pin(L._LOOP_ACTION_CONFIG.stable_motion)
                    )
                },
            },
        ),
    )

    result = L._validate_actual_vendor_authority(
        tmp_path,
        "c" * 40,
        _bundle(),
        VENDOR_CONTRACT_SHA,
    )
    assert [name for name, _value in calls] == ["load", "validate"]
    load_kwargs = calls[0][1]
    assert (
        load_kwargs["expected_receipt_sha256"]
        == L._LOOP_ACTION_CONFIG.runtime_authority_receipt.sha256
    )
    assert (
        load_kwargs["expected_runtime_training_contract_sha256"]
        == VENDOR_CONTRACT_SHA
    )
    assert load_kwargs["launch_commit"] == "c" * 40
    assert result["runtime_plant_identity"] == {"full_candidate_plant": True}

    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.v2.json", "sha256": "b" * 64},
            {
                "schema_version": 2,
                "kind": "agibot_a3_action_dynamic_ready_candidate_v2",
                "action_id": "bh_block",
                "sources": {
                    "stable_motion": {"sha256": "e" * 64}
                },
            },
        ),
    )
    with pytest.raises(L.LaunchRefused, match="action-specific authority"):
        L._validate_actual_vendor_authority(
            tmp_path,
            "c" * 40,
            _bundle(),
            VENDOR_CONTRACT_SHA,
        )


def test_host_plan_binds_vendor_profile_and_single_gpu_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _spec(tmp_path, seed=0, stage="probe")
    document["gpu"]["index"] = 2
    document["gpu"]["lock_path"] = "/tmp/hope_lean_queue_gpu2.lock"
    spec_path = tmp_path / "run.json"
    spec_path.write_bytes(_canonical(document))

    monkeypatch.setattr(
        L._B,
        "_verify_clean_source",
        lambda checkout, commit, **kwargs: {
            "checkout": str(checkout),
            "commit_sha": commit,
        },
    )
    monkeypatch.setattr(
        L._B,
        "_validate_runtime_sources",
        lambda checkout, commit, **kwargs: {
            "N1 vendor diagnostic launcher": {
                "path": L.LAUNCHER_SOURCE,
                "sha256": "1" * 64,
            },
            "N1 diagnostic safety base": {
                "path": L.BASE_LAUNCHER_SOURCE,
                "sha256": "2" * 64,
            },
            f"immutable task profile {L.TASK_PROFILE_ID}": {
                "path": L.TASK_PROFILE_SOURCE,
                "sha256": "3" * 64,
            },
            "vendor A3 robot source": {
                "path": L.ROBOT_SOURCE,
                "sha256": "4" * 64,
            },
            "vendor runtime training-contract identity manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "vendor runtime authority validator": {
                "path": L.VENDOR_AUTHORITY_MODULE_SOURCE,
                "sha256": "5" * 64,
            },
        },
    )
    monkeypatch.setattr(
        L._B,
        "_validate_runtime_asset_environment",
        lambda: {"fixture": True},
    )
    monkeypatch.setattr(
        L._B,
        "_validate_bundle",
        lambda *args, **kwargs: copy.deepcopy(_bundle()),
    )
    monkeypatch.setattr(
        L._B, "_check_rsl_namespace_available", lambda *args: None
    )
    integrated_probe_runtime_sources = {
        "IsaacLab interval event manager": {
            "path": L.INTEGRATED_PROBE_RUNTIME_SOURCE_PINS[
                "IsaacLab interval event manager"
            ]["path"],
            "sha256": L.INTEGRATED_PROBE_RUNTIME_SOURCE_PINS[
                "IsaacLab interval event manager"
            ]["sha256"],
        },
        "IsaacLab push-by-velocity event": {
            "path": L.INTEGRATED_PROBE_RUNTIME_SOURCE_PINS[
                "IsaacLab push-by-velocity event"
            ]["path"],
            "sha256": L.INTEGRATED_PROBE_RUNTIME_SOURCE_PINS[
                "IsaacLab push-by-velocity event"
            ]["sha256"],
        },
    }
    monkeypatch.setattr(
        L,
        "_validate_integrated_probe_runtime_sources",
        lambda: copy.deepcopy(integrated_probe_runtime_sources),
    )
    validated_receipts = {
        "fixed_domain": {"content_sha256": "9" * 64},
        "reward_ppo_economy": {"content_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        L,
        "_validate_prelaunch_receipts",
        lambda *args, **kwargs: copy.deepcopy(validated_receipts),
    )
    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda checkout, commit, **kwargs: {
            "manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "runtime_training_contract_sha256": VENDOR_CONTRACT_SHA,
        },
    )
    monkeypatch.setattr(
        L,
        "_validate_actual_vendor_authority",
        lambda checkout, commit, bundle, contract_sha, **kwargs: {
            "receipt_path": "authority.json",
            "receipt_sha256": "6" * 64,
            "runtime_training_contract": {
                "path": "training_contract.json",
                "sha256": contract_sha,
                "schema_version": 3,
            },
            "runtime_plant_identity": {"fixture": True},
        },
    )
    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            _loop_dynamic_artifact(contract_sha=None),
        ),
    )
    monkeypatch.setattr(
        L._B,
        "_verify_tracked_file",
        lambda _checkout, _commit, pin, **_kwargs: (
            dict(pin),
            tmp_path / pin["path"],
        ),
    )
    with pytest.raises(L.LaunchRefused, match="legacy bundle refused"):
        L.build_plan(spec_path)

    monkeypatch.setattr(
        L._B,
        "_load_tracked_json",
        lambda *args, **kwargs: (
            {"path": "ready.json", "sha256": "f" * 64},
            {
                **_loop_dynamic_artifact(contract_sha=VENDOR_CONTRACT_SHA)
            },
        ),
    )

    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda checkout, commit, **kwargs: {
            "manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "runtime_training_contract_sha256": "8" * 64,
        },
    )
    with pytest.raises(L.LaunchRefused, match="tracked authority"):
        L.build_plan(spec_path)

    monkeypatch.setattr(
        L,
        "_validate_vendor_identity_manifest",
        lambda checkout, commit, **kwargs: {
            "manifest": {
                "path": L.VENDOR_IDENTITY_MANIFEST_SOURCE,
                "sha256": L.VENDOR_IDENTITY_MANIFEST_SHA256,
            },
            "runtime_training_contract_sha256": VENDOR_CONTRACT_SHA,
        },
    )

    plan = L.build_plan(spec_path)
    payload = plan["canonical_payload"]
    assert plan["kind"] == L.CLAIM_KIND
    assert payload["kind"] == L.CLAIM_KIND
    assert payload["spec"]["gpu"]["index"] == 2
    assert payload["spec"]["gpu"]["require_empty"] is True
    assert payload["spec"]["stage"] == "probe"
    assert payload["spec"]["num_envs"] == 4096
    assert payload["spec"]["max_iterations"] == 5
    assert payload["spec"]["save_interval"] == 1
    assert (
        payload["spec"][L.VENDOR_CONTRACT_FIELD]
        == VENDOR_CONTRACT_SHA
    )
    assert f"task={L.TASK_PROFILE_ID}" in payload["training_argv"]
    assert (
        payload["training_argv"].count(L.STABLE_READY_PLANT_OVERRIDE) == 1
    )
    assert (
        payload["training_argv"].count(
            L.VENDOR_DIAGNOSTIC_STAGE_ARG_PREFIX + "probe"
        )
        == 1
    )
    assert (
        payload[L.INTEGRATED_PROBE_CLAIM_FIELD]
        == integrated_probe_runtime_sources
    )
    assert payload[L.PRELAUNCH_RECEIPTS_CLAIM_FIELD] == validated_receipts
    assert payload["formal_evidence_prohibited"] is True
    assert payload["curriculum_promotion_prohibited"] is True
    assert payload["vendor_runtime_authority"]["receipt_sha256"] == "6" * 64
    assert plan["launch_claim_sha256"] == L.canonical_sha256(payload)
