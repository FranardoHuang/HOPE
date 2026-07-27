from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_MODULE = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "actor_observation_contract.py"
)
_SPEC = importlib.util.spec_from_file_location("task_first_actor_contract", _MODULE)
assert _SPEC is not None and _SPEC.loader is not None
contract_mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contract_mod
_SPEC.loader.exec_module(contract_mod)


@pytest.mark.parametrize("action_count", [1, 2, 5, 6, 93])
def test_task_first_contract_is_explicitly_sized(action_count):
    contract = contract_mod.resolve_actor_observation_contract(
        f"task_first_n{action_count}"
    )
    assert contract.name == f"task_first_n{action_count}"
    assert contract.obs_mode == "hitter_footwork"
    assert contract.total_dim == 181 + action_count
    assert contract.layout[-2:] == (
        ("racket_target_normal_cmd", 4),
        ("action_one_hot", action_count),
    )


def test_action_bank_size_changes_the_contract_and_shape():
    five = contract_mod.resolve_actor_observation_contract("task_first_n5")
    six = contract_mod.resolve_actor_observation_contract("task_first_n6")
    assert five.name != six.name
    assert five.total_dim + 1 == six.total_dim
    assert five.layout[-1] == ("action_one_hot", 5)
    assert six.layout[-1] == ("action_one_hot", 6)


@pytest.mark.parametrize(
    "name",
    ["task_first_n0", "task_first_n-1", "task_first_n", "task_first_n1.5"],
)
def test_invalid_dynamic_contract_names_fail_closed(name):
    with pytest.raises(ValueError, match="Unknown actor observation contract"):
        contract_mod.resolve_actor_observation_contract(name)
