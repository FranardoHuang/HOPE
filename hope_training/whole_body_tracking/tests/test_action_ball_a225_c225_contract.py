"""Exact contract-only tests for the fixed-midpoint N1 A225/C225 split."""

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
        "action_ball_a225_c225_contract", CONTRACT_PATH
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


def test_a225_and_c225_are_distinct_registered_exact_width_contracts():
    module = _load_contract_module()
    a225 = module.resolve_actor_observation_contract("action_ball_a225")
    c225 = module.resolve_actor_observation_contract("action_ball_c225")

    assert a225 is module.ACTION_BALL_A225
    assert c225 is module.ACTION_BALL_C225
    assert a225 is not c225
    assert a225.name == a225.obs_mode == "action_ball_a225"
    assert c225.name == c225.obs_mode == "action_ball_c225"
    assert a225.total_dim == c225.total_dim == 225
    assert sum(term.dim for term in a225.terms) == 225
    assert sum(term.dim for term in c225.terms) == 225


def test_a225_c225_share_exact_historical_prefix_and_station_clock_suffix():
    module = _load_contract_module()
    historical = module.STAGE1_NATURAL_CLIP_PADDLE_WORLD_V2
    a225 = module.ACTION_BALL_A225
    c225 = module.ACTION_BALL_C225

    assert sum(term.dim for term in historical.terms[:10]) == 212
    assert a225.terms[:10] == c225.terms[:10] == historical.terms[:10]
    assert a225.terms[-3:] == c225.terms[-3:] == historical.terms[-3:]

    common_expected = {
        "desired_base_xy_world": (221, 223),
        "time_to_contact": (223, 224),
        "time_to_teacher_start": (224, 225),
    }
    for contract in (a225, c225):
        slices = _slices(contract)
        for name, expected in common_expected.items():
            assert slices[name] == expected


def test_a225_212_to_221_is_only_task_desired_contact_p_v_face():
    module = _load_contract_module()
    contract = module.ACTION_BALL_A225
    expected = (
        ("task_desired_contact_position_heading", 3),
        ("task_desired_contact_velocity_heading", 3),
        ("task_desired_contact_face_heading", 3),
    )
    assert contract.layout[10:13] == expected
    assert tuple(_slices(contract)[name] for name, _dim in expected) == (
        (212, 215),
        (215, 218),
        (218, 221),
    )
    assert {
        term.deploy_source for term in contract.terms[10:13]
    } == {"action_ball_a225_atomic_desired_contact_snapshot"}


def test_c225_212_to_221_is_only_incoming_ball_p_v_spin():
    module = _load_contract_module()
    contract = module.ACTION_BALL_C225
    expected = (
        ("incoming_ball_contact_position_heading", 3),
        ("incoming_ball_contact_velocity_heading", 3),
        ("incoming_ball_contact_spin_heading", 3),
    )
    assert contract.layout[10:13] == expected
    assert tuple(_slices(contract)[name] for name, _dim in expected) == (
        (212, 215),
        (215, 218),
        (218, 221),
    )
    assert {
        term.deploy_source for term in contract.terms[10:13]
    } == {"action_ball_c225_atomic_causal_question_snapshot"}

    task_block_text = " ".join(
        f"{term.name} {term.deploy_source} {term.description}"
        for term in contract.terms[10:13]
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
    assert "def action_ball_c225_incoming_ball_contact_position_heading" in observation_source
    c_source_tuple = observation_source.split('"c225": (', 1)[1].split("),", 1)[0]
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
        (module.ACTION_BALL_A225, "HOPEActionBallA225ObservationsCfg"),
        (module.ACTION_BALL_C225, "HOPEActionBallC225ObservationsCfg"),
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
        assert sum(dim for _name, dim in contract.layout) == 225
