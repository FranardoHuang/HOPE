"""Host-only tests for effective reward recipe receipts."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from collections import OrderedDict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "source/whole_body_tracking/whole_body_tracking/utils"
    / "effective_reward_recipe.py"
)
SPEC = importlib.util.spec_from_file_location("effective_reward_recipe_under_test", MODULE_PATH)
RECIPE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RECIPE)


def racket_position_reward():
    pass


def racket_velocity_reward():
    pass


def racket_normal_reward():
    pass


def alternate_position_reward():
    pass


@dataclasses.dataclass
class Term:
    func: object
    weight: float
    params: object = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Selector:
    name: str
    body_names: list
    body_ids: slice = dataclasses.field(default_factory=lambda: slice(None))


class ImplicitActuatorCfg:
    pass


class IdealPDActuatorCfg:
    pass


@dataclasses.dataclass
class RobotCfg:
    actuators: dict


@dataclasses.dataclass
class SceneCfg:
    robot: RobotCfg


@dataclasses.dataclass
class EnvironmentCfg:
    rewards: dict
    scene: SceneCfg


def _quality_cfg(weights):
    funcs = {
        "racket_position": racket_position_reward,
        "racket_velocity": racket_velocity_reward,
        "racket_normal": racket_normal_reward,
    }
    return {
        "task": {
            "rewards": {
                name: Term(funcs[name], weights[index], {"command_name": "racket_target"})
                for index, name in enumerate(funcs)
            }
        }
    }


def test_live_yaml_quality_weights_and_nominal_pack_have_different_sha():
    # The effective values currently supplied by task YAML are not the v2 pack's
    # nominal frozen values.  A receipt makes that difference machine-checkable.
    yaml_effective = _quality_cfg((4.0, 0.5, 0.5))
    nominal_pack = _quality_cfg((393.4, 295.1, 229.5))

    yaml_receipt = RECIPE.build_effective_reward_receipt(yaml_effective)
    pack_receipt = RECIPE.build_effective_reward_receipt(nominal_pack)

    assert yaml_receipt["sha256"] != pack_receipt["sha256"]
    assert {term["name"]: term["weight"] for term in yaml_receipt["terms"]} == {
        "racket_position": 4.0,
        "racket_velocity": 0.5,
        "racket_normal": 0.5,
    }
    assert {term["name"]: term["weight"] for term in pack_receipt["terms"]} == {
        "racket_position": 393.4,
        "racket_velocity": 295.1,
        "racket_normal": 229.5,
    }


def test_term_and_param_order_do_not_change_canonical_json_or_sha():
    term_a = Term(
        racket_position_reward,
        4.0,
        OrderedDict((("std", 0.2), ("command_name", "racket_target"))),
    )
    term_b = Term(racket_velocity_reward, 0.5, {"command_name": "racket_target"})
    left = {"rewards": OrderedDict((("z_term", term_b), ("a_term", term_a)))}

    term_a_reordered = Term(
        racket_position_reward,
        4.0,
        OrderedDict((("command_name", "racket_target"), ("std", 0.2))),
    )
    right = {"rewards": OrderedDict((("a_term", term_a_reordered), ("z_term", term_b)))}

    left_recipe = RECIPE.effective_reward_recipe(left)
    right_recipe = RECIPE.effective_reward_recipe(right)
    assert [term["name"] for term in left_recipe["terms"]] == ["a_term", "z_term"]
    assert RECIPE.canonical_effective_reward_recipe_json(left_recipe) == (
        RECIPE.canonical_effective_reward_recipe_json(right_recipe)
    )
    assert RECIPE.effective_reward_recipe_sha256(left) == (
        RECIPE.effective_reward_recipe_sha256(right)
    )


@pytest.mark.parametrize(
    "changed",
    [
        Term(racket_position_reward, 5.0, {"std": 0.2}),
        Term(racket_position_reward, 4.0, {"std": 0.3}),
        Term(alternate_position_reward, 4.0, {"std": 0.2}),
    ],
    ids=("weight", "params", "callable"),
)
def test_weight_params_and_callable_changes_each_change_sha(changed):
    baseline = {"rewards": {"position": Term(racket_position_reward, 4.0, {"std": 0.2})}}
    variant = {"rewards": {"position": changed}}
    assert RECIPE.effective_reward_recipe_sha256(baseline) != (
        RECIPE.effective_reward_recipe_sha256(variant)
    )


def test_inactive_none_and_zero_weight_terms_are_omitted():
    cfg = {
        "rewards": {
            "none_term": None,
            "zero_term": Term(racket_velocity_reward, 0.0, {"ignored": object()}),
            "active_term": Term(racket_position_reward, 4.0),
        }
    }
    receipt = RECIPE.build_effective_reward_receipt(cfg)
    assert [term["name"] for term in receipt["terms"]] == ["active_term"]


def _backend_cfg(actuator_type):
    return EnvironmentCfg(
        rewards={
            "arm_torque_saturation": Term(
                racket_velocity_reward,
                -0.5,
                {"command_name": "racket_target"},
            ),
            "active_term": Term(racket_position_reward, 4.0),
        },
        scene=SceneCfg(
            robot=RobotCfg(
                actuators={
                    "arms": actuator_type(),
                    "waist": actuator_type(),
                }
            )
        ),
    )


def test_implicit_arm_torque_request_has_explicit_disabled_receipt():
    cfg = _backend_cfg(ImplicitActuatorCfg)

    compatibility = RECIPE.build_reward_backend_compatibility_receipt(cfg)
    assert set(compatibility) == {
        "schema_version",
        "kind",
        "effective_reward_recipe_sha256",
        "decisions",
        "sha256",
    }
    assert compatibility["schema_version"] == 1
    assert (
        compatibility["kind"]
        == "whole_body_tracking.reward_backend_compatibility"
    )
    assert compatibility["decisions"] == [
        {
            "name": "arm_torque_saturation",
            "status": "disabled_incompatible_actuator_backend",
            "requested_weight": -0.5,
            "effective_weight": 0.0,
            "reason_code": (
                "implicit_actuator_has_no_proven_explicit_preclip_demand"
            ),
            "reason": (
                "ImplicitActuator does not expose a proven explicit "
                "pre-clip demand through computed_torque"
            ),
            "actuator_backends": {
                "arms": "implicit",
                "waist": "implicit",
            },
        }
    ]
    payload = {
        "schema_version": compatibility["schema_version"],
        "kind": compatibility["kind"],
        "effective_reward_recipe_sha256": compatibility[
            "effective_reward_recipe_sha256"
        ],
        "decisions": compatibility["decisions"],
    }
    assert compatibility["sha256"] == hashlib.sha256(
        RECIPE.canonical_reward_backend_compatibility_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()
    assert cfg.rewards["arm_torque_saturation"].weight == 0.0

    # The active recipe and activation identity remain truthful: the disabled
    # request is evidence in the compatibility receipt, not an active term.
    effective = RECIPE.build_effective_reward_receipt(cfg)
    assert (
        compatibility["effective_reward_recipe_sha256"]
        == effective["sha256"]
    )
    assert [term["name"] for term in effective["terms"]] == ["active_term"]
    baseline = _backend_cfg(ImplicitActuatorCfg)
    baseline.rewards["arm_torque_saturation"].weight = 0.0
    assert effective["sha256"] == RECIPE.effective_reward_recipe_sha256(
        baseline
    )


def test_mapping_backed_implicit_arm_torque_request_is_disabled_truthfully():
    cfg = {
        "rewards": {
            "arm_torque_saturation": {
                "func": racket_velocity_reward,
                "weight": -0.5,
                "params": {"command_name": "racket_target"},
            },
            "active_term": Term(racket_position_reward, 4.0),
        },
        "scene": {
            "robot": {
                "actuators": {
                    "arms": ImplicitActuatorCfg(),
                    "waist": ImplicitActuatorCfg(),
                }
            }
        },
    }

    compatibility = RECIPE.build_reward_backend_compatibility_receipt(cfg)

    assert compatibility["decisions"][0]["requested_weight"] == -0.5
    assert compatibility["decisions"][0]["effective_weight"] == 0.0
    assert cfg["rewards"]["arm_torque_saturation"]["weight"] == 0.0
    assert [
        term["name"]
        for term in RECIPE.build_effective_reward_receipt(cfg)["terms"]
    ] == ["active_term"]


def test_explicit_arm_torque_request_remains_effective_and_is_not_disabled():
    cfg = _backend_cfg(IdealPDActuatorCfg)

    compatibility = RECIPE.build_reward_backend_compatibility_receipt(cfg)
    assert compatibility["decisions"][0] == {
        "name": "arm_torque_saturation",
        "status": "enabled_compatible_actuator_backend",
        "requested_weight": -0.5,
        "effective_weight": -0.5,
        "reason_code": "explicit_actuator_preclip_demand_available",
        "reason": (
            "explicit actuator backend exposes the pre-clip demand required "
            "by arm_torque_saturation"
        ),
        "actuator_backends": {
            "arms": "explicit",
            "waist": "explicit",
        },
    }
    effective = RECIPE.build_effective_reward_receipt(cfg)
    assert {
        term["name"]: term["weight"] for term in effective["terms"]
    }["arm_torque_saturation"] == -0.5


def test_expected_sha_passes_exactly_and_mismatch_is_rejected():
    cfg = _quality_cfg((4.0, 0.5, 0.5))
    expected = RECIPE.effective_reward_recipe_sha256(cfg)
    assert RECIPE.build_effective_reward_receipt(cfg, expected_sha256=expected)["sha256"] == expected

    with pytest.raises(RECIPE.RewardRecipeMismatchError, match="expected .* got"):
        RECIPE.build_effective_reward_receipt(cfg, expected_sha256="0" * 64)


def test_receipt_payload_is_valid_stable_json_with_callable_identity():
    cfg = {"rewards": {"position": Term(racket_position_reward, 4, {"std": 0.2})}}
    recipe = RECIPE.effective_reward_recipe(cfg)
    encoded = RECIPE.canonical_effective_reward_recipe_json(recipe)
    assert json.loads(encoded) == recipe
    assert recipe["terms"][0]["callable"].endswith(
        "test_effective_reward_recipe.racket_position_reward"
    )
    assert recipe["terms"][0]["weight"] == 4.0


def test_dataclass_config_params_and_slice_are_serialized_explicitly():
    selector = Selector("robot", ["left_foot", "right_foot"])
    cfg = {
        "rewards": {
            "contact": Term(
                racket_normal_reward,
                -0.5,
                {"asset_cfg": selector},
            )
        }
    }
    params = RECIPE.effective_reward_recipe(cfg)["terms"][0]["params"]
    serialized_selector = params["asset_cfg"]
    assert serialized_selector["__config_type__"].endswith(
        "test_effective_reward_recipe.Selector"
    )
    assert serialized_selector["fields"]["body_ids"] == {
        "__slice__": [None, None, None]
    }


@pytest.mark.parametrize(
    "bad_params",
    [
        {"opaque": object()},
        {"not_finite": float("nan")},
        {"unordered": {"left", "right"}},
        {1: "non-string key"},
    ],
    ids=("opaque-object", "nan", "set", "non-string-key"),
)
def test_unstable_or_non_json_params_fail_closed(bad_params):
    cfg = {"rewards": {"position": Term(racket_position_reward, 4.0, bad_params)}}
    with pytest.raises(RECIPE.RewardRecipeError):
        RECIPE.build_effective_reward_receipt(cfg)


def test_anonymous_or_stateful_callable_fails_closed():
    with pytest.raises(RECIPE.RewardRecipeError, match="stable module-qualified"):
        RECIPE.build_effective_reward_receipt(
            {"rewards": {"position": Term(lambda: None, 4.0)}}
        )

    class StatefulCallable:
        def __call__(self):
            pass

    with pytest.raises(RECIPE.RewardRecipeError, match="named function"):
        RECIPE.build_effective_reward_receipt(
            {"rewards": {"position": Term(StatefulCallable(), 4.0)}}
        )
