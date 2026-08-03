"""Exact contract-only tests for the fresh fixed-midpoint A211/C211 split."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "actor_observation_contract.py"
)
OBSERVATION_PATH = CONTRACT_PATH.parent / "mdp" / "hope_observations.py"
ENV_CFG_PATH = (
    CONTRACT_PATH.parent
    / "config"
    / "agibot_a3"
    / "hope_env_cfg.py"
)


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "action_ball_a211_c211_contract", CONTRACT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _slices(contract) -> dict[str, tuple[int, int]]:
    result = {}
    offset = 0
    for term in contract.terms:
        result[term.name] = (offset, offset + term.dim)
        offset += term.dim
    assert offset == contract.total_dim
    return result


def test_a211_and_c211_are_distinct_registered_exact_width_contracts():
    module = _load_contract_module()
    a211 = module.resolve_actor_observation_contract("action_ball_a211")
    c211 = module.resolve_actor_observation_contract("action_ball_c211")

    assert a211 is module.ACTION_BALL_A211
    assert c211 is module.ACTION_BALL_C211
    assert a211 is not c211
    assert a211.name == a211.obs_mode == "action_ball_a211"
    assert c211.name == c211.obs_mode == "action_ball_c211"
    assert a211.total_dim == c211.total_dim == 211
    assert sum(term.dim for term in a211.terms) == 211
    assert sum(term.dim for term in c211.terms) == 211


def test_a211_c211_remove_only_actor_teacher_base_and_append_validity():
    module = _load_contract_module()
    historical = module.STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2
    a211 = module.ACTION_BALL_A211
    c211 = module.ACTION_BALL_C211

    assert sum(term.dim for term in historical.terms[:10]) == 212
    expected_prefix = historical.terms[:1] + historical.terms[2:10]
    assert a211.terms[:9] == c211.terms[:9] == expected_prefix
    assert all(term.name != "teacher_base_now_world" for term in a211.terms)
    assert all(term.name != "teacher_base_now_world" for term in c211.terms)
    assert a211.terms[-4:-1] == c211.terms[-4:-1] == historical.terms[-3:]
    assert a211.layout[-1] == c211.layout[-1] == ("task_valid", 1)

    common_expected = {
        "desired_base_xy_world": (206, 208),
        "time_to_contact": (208, 209),
        "time_to_teacher_start": (209, 210),
        "task_valid": (210, 211),
    }
    for contract in (a211, c211):
        slices = _slices(contract)
        for name, expected in common_expected.items():
            assert slices[name] == expected


def test_a211_197_to_206_is_only_task_desired_contact_p_v_face():
    module = _load_contract_module()
    contract = module.ACTION_BALL_A211
    expected = (
        ("task_desired_contact_position_heading", 3),
        ("task_desired_contact_velocity_heading", 3),
        ("task_desired_contact_face_heading", 3),
    )
    assert contract.layout[9:12] == expected
    assert tuple(_slices(contract)[name] for name, _dim in expected) == (
        (197, 200),
        (200, 203),
        (203, 206),
    )
    assert {
        term.deploy_source for term in contract.terms[9:12]
    } == {"action_ball_a211_atomic_desired_contact_snapshot"}


def test_c211_197_to_206_is_only_incoming_ball_p_v_spin():
    module = _load_contract_module()
    contract = module.ACTION_BALL_C211
    expected = (
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
    )
    assert contract.layout[9:12] == expected
    assert tuple(_slices(contract)[name] for name, _dim in expected) == (
        (197, 200),
        (200, 203),
        (203, 206),
    )
    assert {
        term.deploy_source for term in contract.terms[9:12]
    } == {"action_ball_c211_atomic_causal_question_snapshot"}

    task_block_text = " ".join(
        f"{term.name} {term.deploy_source} {term.description}"
        for term in contract.terms[9:12]
    ).lower()
    for forbidden in (
        "desired",
        "racket",
        "planner_contact",
        "current_lm",
        "landing",
        "table midpoint",
        "validity",
        "estimate_age",
    ):
        assert forbidden not in task_block_text


def test_c_producer_never_reads_fixed_midpoint_or_solved_contact_target():
    observation_source = OBSERVATION_PATH.read_text(encoding="utf-8")
    assert "def action_ball_c211_incoming_ball_contact_position_heading" in observation_source
    c_source_tuple = observation_source.split('"c211": (', 1)[1].split("),", 1)[0]
    assert "_action_ball_ball_contact_target_w" in c_source_tuple
    assert "vb_vel_in_w" in c_source_tuple
    assert "vb_spin_in_w" in c_source_tuple
    for forbidden in (
        "racket_target",
        "desired",
        "landing",
        "_vb_target_xy_per_env",
        "current_lm",
    ):
        assert forbidden not in c_source_tuple


def test_registered_contract_layouts_match_policy_config_term_order_exactly():
    module = _load_contract_module()
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENV_CFG_PATH))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    pairs = (
        (module.ACTION_BALL_A211, "HOPEActionBallA211ObservationsCfg"),
        (module.ACTION_BALL_C211, "HOPEActionBallC211ObservationsCfg"),
    )
    for contract, outer_name in pairs:
        policy = next(
            child
            for child in classes[outer_name].body
            if isinstance(child, ast.ClassDef)
        )
        config_names = tuple(
            target.id
            for child in policy.body
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call)
            for target in child.targets
            if isinstance(target, ast.Name)
        )
        assert config_names == tuple(name for name, _dim in contract.layout)
        assert sum(dim for _name, dim in contract.layout) == 211


def test_historical_h225_and_a225_c225_receipts_are_not_relabelled():
    module = _load_contract_module()
    assert module.STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2.total_dim == 225
    assert module.ACTION_BALL_A225.name == "action_ball_a225"
    assert module.ACTION_BALL_C225.name == "action_ball_c225"
    assert module.ACTION_BALL_A225.total_dim == module.ACTION_BALL_C225.total_dim == 225


def test_a211_c211_envs_pin_the_frozen_wait_schedule_and_runtime_checks():
    source = ENV_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENV_CFG_PATH))
    classes = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    expected_assignments = (
        "action_ball_task_wait_enabled = True",
        "action_ball_task_wait_policy_dt_s = 0.02",
        "action_ball_task_wait_seed = 20260804",
        "action_ball_task_wait_min_wait_ticks = 5",
        "action_ball_task_wait_max_wait_ticks = 25",
        "action_ball_task_wait_episode_horizon_ticks = 500",
        "action_ball_task_wait_required_active_ticks = 200",
    )
    for name in (
        "HOPEPingPongActionBallA211AgibotA3EnvCfg",
        "HOPEPingPongActionBallC211AgibotA3EnvCfg",
    ):
        segment = ast.get_source_segment(source, classes[name])
        assert segment is not None
        for assignment in expected_assignments:
            assert assignment in segment
        assert "_validate_action_ball_211_wait_schedule_cfg(self)" in segment

    assert "sim.dt * decimation == 0.02 s" in source
    assert "requires a 500-policy-tick episode horizon" in source
