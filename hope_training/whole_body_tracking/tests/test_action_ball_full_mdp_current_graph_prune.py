"""Current-graph proof for retired FullMDP gates and legacy-only modules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "source" / "whole_body_tracking"
TRACKING = (
    SOURCE_ROOT
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
)
MDP = TRACKING / "mdp"
CURRENT_MDP_INIT = MDP / "__init__.py"
CURRENT_ENV_CFG = TRACKING / "config" / "agibot_a3" / "hope_env_cfg.py"
CURRENT_TERMINATIONS = MDP / "terminations.py"
REVEAL_BOUNDARY = SOURCE_ROOT / "action_ball_full_mdp_reveal_boundary.py"
LEAN_OBSERVATIONS = MDP / "action_ball_full_mdp_lean_observation_cfg.py"

LEGACY_MODULES = {
    "action_ball_full_mdp_rewards",
    "action_ball_full_mdp_terminations",
}
RETIRED_CURRENT_SYMBOLS = {
    "fresh_pre_reward_publish",
    "fresh_full_mdp_pre_reward_done_term",
}


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(path: Path) -> set[str]:
    return {
        node.module or ""
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom)
    }


def _called_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def test_current_ast_and_callsites_exclude_retired_modules_and_gate():
    assert LEGACY_MODULES.isdisjoint(_imported_modules(CURRENT_MDP_INIT))
    for path in (CURRENT_ENV_CFG, CURRENT_TERMINATIONS):
        source = path.read_text(encoding="utf-8")
        assert RETIRED_CURRENT_SYMBOLS.isdisjoint(_called_names(path))
        assert RETIRED_CURRENT_SYMBOLS.isdisjoint(
            node.id for node in ast.walk(_tree(path)) if isinstance(node, ast.Name)
        )
        assert "fresh_pre_reward_publish" not in source
        assert "fresh_full_mdp_pre_reward_done_term" not in source


def test_removed_no_consumer_counters_are_absent_from_current_sources():
    assert "minted_receipt_count" not in REVEAL_BOUNDARY.read_text(encoding="utf-8")
    assert "_minted_receipt_count" not in REVEAL_BOUNDARY.read_text(encoding="utf-8")
    assert "semantic_publication_count" not in LEAN_OBSERVATIONS.read_text(
        encoding="utf-8"
    )
    assert "_semantic_publications" not in LEAN_OBSERVATIONS.read_text(
        encoding="utf-8"
    )


def _require_live_isaac_import_surface() -> None:
    pytest.importorskip("isaaclab")
    pytest.importorskip("warp")
    pytest.importorskip("omni.kit.app")


def test_legacy_modules_remain_explicitly_importable_but_not_current_exports():
    _require_live_isaac_import_surface()
    current = importlib.import_module("whole_body_tracking.tasks.tracking.mdp")
    assert not hasattr(current, "FreshFullMdpRewardGraph")
    assert not hasattr(current, "fresh_full_mdp_pre_reward_done_term")

    legacy_rewards = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_rewards"
    )
    legacy_terminations = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_full_mdp_terminations"
    )
    assert callable(legacy_rewards.construct_diagnostic_fresh_full_mdp_reward_graph)
    assert callable(legacy_terminations.fresh_full_mdp_pre_reward_done_term)
