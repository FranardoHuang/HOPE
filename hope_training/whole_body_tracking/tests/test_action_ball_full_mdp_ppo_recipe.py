"""Dependency-free checks for the shared continuous FullMDP PPO V2 recipe."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "action_ball_full_mdp_ppo_recipe.py"
)


def _load():
    name = "action_ball_full_mdp_ppo_recipe_test"
    spec = importlib.util.spec_from_file_location(name, RECIPE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_recipe_is_frozen_complete_and_keeps_total_training_work():
    recipe = _load().ACTION_BALL_FULL_MDP_PPO_RECIPE
    assert recipe.kind == "action_ball_full_mdp_ppo_v2"
    assert (
        recipe.num_steps_per_env,
        recipe.max_iterations,
        recipe.save_interval,
    ) == (48, 12_500, 500)
    assert recipe.empirical_normalization is False
    assert (recipe.num_learning_epochs, recipe.num_mini_batches) == (5, 8)
    assert (recipe.gamma, recipe.lam) == (0.99, 0.98)
    assert 48 * 12_500 == 24 * 25_000
    assert 5 * 8 * 12_500 == 5 * 4 * 25_000
    assert 48 * 500 == 24 * 1000
    with pytest.raises(FrozenInstanceError):
        recipe.lam = 0.95


def test_learning_identity_matches_existing_training_contract_serializer_shape():
    recipe = _load().ACTION_BALL_FULL_MDP_PPO_RECIPE
    scientific = recipe.learning_recipe()
    assert scientific["runner"] == {
        "num_steps_per_env": 48,
        "empirical_normalization": False,
    }
    assert scientific["policy"] == {
        "class_name": "ActorCritic",
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "init_noise_std": 0.02,
        "noise_std_type": "log",
    }
    assert scientific["algorithm"]["num_mini_batches"] == 8
    assert scientific["algorithm"]["lam"] == 0.98
    payload = json.dumps(
        scientific,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert recipe.learning_recipe_sha256() == hashlib.sha256(payload).hexdigest()


def test_isaac_and_mujoco_views_share_scientific_values_without_aliasing():
    recipe = _load().ACTION_BALL_FULL_MDP_PPO_RECIPE
    isaac = recipe.isaac_overrides()
    mujoco = recipe.mujoco_train_cfg()
    assert isaac["runner"] == {
        "num_steps_per_env": 48,
        "max_iterations": 12_500,
        "save_interval": 500,
        "empirical_normalization": False,
    }
    assert mujoco["num_steps_per_env"] == isaac["runner"]["num_steps_per_env"]
    assert mujoco["save_interval"] == isaac["runner"]["save_interval"]
    assert mujoco["policy"]["actor_obs_normalization"] is False
    assert mujoco["policy"]["critic_obs_normalization"] is False
    assert {
        key: mujoco["policy"][key] for key in isaac["policy"]
    } == isaac["policy"]
    assert mujoco["algorithm"] == isaac["algorithm"]
    isaac["runner"]["num_steps_per_env"] = 1
    assert recipe.num_steps_per_env == 48
