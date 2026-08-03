"""Dependency-light contract tests for the diagnostic MuJoCo trainer shell."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


T = importlib.import_module("hope_training.whole_body_tracking.mujoco_native.trainer")
C = importlib.import_module(
    "hope_training.whole_body_tracking.mujoco_native.checkpoint"
)
ROOT = Path(__file__).resolve().parents[1]
TRAINER_SOURCE = ROOT / "mujoco_native/trainer.py"
CHECKPOINT_SOURCE = ROOT / "mujoco_native/checkpoint.py"


def _digest(character: str) -> str:
    return character * 64


def _identity():
    return T.TrainerIdentity(
        contract_sha256=_digest("a"),
        observation_contract_sha256=_digest("b"),
        action_contract_sha256=_digest("c"),
        reward_contract_sha256=_digest("d"),
    )


def _receipt(identity, **overrides):
    value = {
        "kind": T.DIAGNOSTIC_TRAINER_RECEIPT_KIND,
        "ppo_ready": True,
        "reward_available": True,
        "normal_step_available": True,
        "reset_boundary_checkpoint_available": True,
        "diagnostic_unauthorized": True,
        "formal_authorized": False,
        "mid_episode_resume": False,
        "blockers": [],
        **identity.as_dict(),
    }
    value.update(overrides)
    return value


def test_modules_import_without_eager_torch_or_rsl_rl_dependency():
    trees = [
        ast.parse(TRAINER_SOURCE.read_text(encoding="utf-8")),
        ast.parse(CHECKPOINT_SOURCE.read_text(encoding="utf-8")),
    ]
    eager_imports = set()
    for tree in trees:
        for node in tree.body:
            if isinstance(node, ast.Import):
                eager_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                eager_imports.add(node.module)
    assert "torch" not in eager_imports
    assert all(not name.startswith("rsl_rl") for name in eager_imports)
    assert hasattr(T, "MujocoDiagnosticPPOTrainer")
    assert hasattr(C, "ResetBoundaryCheckpoint")


def test_identity_and_config_are_content_bound_without_runtime_dependencies():
    identity = _identity()
    assert identity.as_dict() == {
        "contract_sha256": _digest("a"),
        "observation_contract_sha256": _digest("b"),
        "action_contract_sha256": _digest("c"),
        "reward_contract_sha256": _digest("d"),
    }
    config = T.DiagnosticPPOConfig(observation_dim=3, action_dim=2)
    assert len(config.content_sha256) == 64
    assert (
        config.content_sha256
        == T.DiagnosticPPOConfig(observation_dim=3, action_dim=2).content_sha256
    )
    with pytest.raises(T.DiagnosticPPOContractError, match="SHA-256"):
        T.TrainerIdentity(
            contract_sha256="bad",
            observation_contract_sha256=_digest("b"),
            action_contract_sha256=_digest("c"),
            reward_contract_sha256=_digest("d"),
        )


def test_readiness_requires_positive_diagnostic_fields_and_all_four_shas():
    identity = _identity()
    accepted = T.validate_diagnostic_readiness_receipt(_receipt(identity), identity)
    assert accepted["diagnostic_unauthorized"] is True
    assert accepted["formal_authorized"] is False
    assert accepted["mid_episode_resume"] is False

    for field, bad in (
        ("ppo_ready", False),
        ("reward_available", False),
        ("normal_step_available", False),
        ("reset_boundary_checkpoint_available", False),
        ("diagnostic_unauthorized", False),
        ("formal_authorized", True),
        ("mid_episode_resume", True),
        ("contract_sha256", _digest("1")),
        ("observation_contract_sha256", _digest("2")),
        ("action_contract_sha256", _digest("3")),
        ("reward_contract_sha256", _digest("4")),
    ):
        with pytest.raises(T.DiagnosticPPOBlocked, match=field):
            T.validate_diagnostic_readiness_receipt(
                _receipt(identity, **{field: bad}), identity
            )


def test_nonempty_blockers_and_existing_vecenv_style_blocked_receipt_fail_closed():
    identity = _identity()
    with pytest.raises(T.DiagnosticPPOBlocked, match="blocked"):
        T.validate_diagnostic_readiness_receipt(
            _receipt(identity, blockers=["reward_contract_missing"]), identity
        )
    with pytest.raises(T.DiagnosticPPOBlocked, match="kind"):
        T.validate_diagnostic_readiness_receipt(
            {
                "reward_available": False,
                "diagnostic_unauthorized": True,
                "blockers": ["full_reward_missing"],
            },
            identity,
        )


def test_checkpoint_schema_explicitly_denies_mid_episode_and_formal_resume():
    source = CHECKPOINT_SOURCE.read_text(encoding="utf-8")
    assert C.CHECKPOINT_KIND.endswith("reset_boundary_checkpoint_v1")
    assert '"kind": "explicit_full_reset_boundary"' in source
    assert '"mid_episode_resume": False' in source
    assert '"diagnostic_unauthorized": True' in source
    assert '"formal_authorized": False' in source
    for state_name in (
        "model_state_dict",
        "optimizer_state_dict",
        "normalizer_state_dict",
        "rng_state",
        "update_counter",
    ):
        assert state_name in source
