"""Current-graph proof for retired FullMDP gates and legacy-only modules."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


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
LIVE_TERMINATION_NAMES = (
    "time_out",
    "base_fell_tilt",
    "base_too_low",
    "joint_qdes_forbidden",
    "robot_hit_table",
)
LIVE_TERMINATION_CALLABLES = (
    "time_out",
    "bad_orientation",
    "root_height_below_minimum",
    "pre_clamp_qdes_forbidden_zone",
    "table_hit_done_term",
)


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


def test_current_config_constructs_exact_five_live_termination_terms():
    tree = _tree(CURRENT_ENV_CFG)
    order = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "ACTION_BALL_FULL_MDP_TERMINATION_MANAGER_ORDER"
            for target in node.targets
        )
    )
    assert isinstance(order, ast.Tuple)
    assert tuple(ast.literal_eval(item) for item in order.elts) == LIVE_TERMINATION_NAMES

    cfg_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "HOPEActionBallFullMdpTerminationsCfg"
    )
    assignments = tuple(
        node
        for node in cfg_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    )
    assert tuple(node.targets[0].id for node in assignments) == LIVE_TERMINATION_NAMES
    assert all(isinstance(node.value, ast.Call) for node in assignments)
    callables = []
    for node in assignments:
        call = node.value
        assert isinstance(call, ast.Call)
        if isinstance(call.func, ast.Name) and call.func.id == "table_hit_done_term":
            callables.append(call.func.id)
            continue
        func_keyword = next(keyword for keyword in call.keywords if keyword.arg == "func")
        assert isinstance(func_keyword.value, ast.Attribute)
        callables.append(func_keyword.value.attr)
    assert tuple(callables) == LIVE_TERMINATION_CALLABLES


def test_removed_no_consumer_counters_are_absent_from_current_sources():
    assert "minted_receipt_count" not in REVEAL_BOUNDARY.read_text(encoding="utf-8")
    assert "_minted_receipt_count" not in REVEAL_BOUNDARY.read_text(encoding="utf-8")
    assert "semantic_publication_count" not in LEAN_OBSERVATIONS.read_text(
        encoding="utf-8"
    )
    assert "_semantic_publications" not in LEAN_OBSERVATIONS.read_text(
        encoding="utf-8"
    )


def _load_explicit(name: str, path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_legacy_modules_remain_explicitly_importable(monkeypatch):
    legacy_rewards = _load_explicit(
        "action_ball_full_mdp_rewards",
        MDP / "action_ball_full_mdp_rewards.py",
        monkeypatch,
    )
    legacy_terminations = _load_explicit(
        "action_ball_full_mdp_terminations",
        MDP / "action_ball_full_mdp_terminations.py",
        monkeypatch,
    )
    assert callable(legacy_rewards.construct_diagnostic_fresh_full_mdp_reward_graph)
    assert callable(legacy_terminations.fresh_full_mdp_pre_reward_done_term)
