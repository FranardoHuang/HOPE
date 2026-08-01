"""Host-only tests for the fresh N1/N5 shared-ready actor bootstrap."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PATH = (
    ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "train.py"
)
CONTRACT_PATH = (
    ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "training_contract.py"
)


class NS(types.SimpleNamespace):
    pass


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def contract_mod():
    return _load_by_path(
        "_action_ball_bootstrap_contract_under_test", CONTRACT_PATH
    )


@pytest.fixture(scope="module")
def train_mod(contract_mod):
    package = sys.modules.setdefault(
        "whole_body_tracking", types.ModuleType("whole_body_tracking")
    )
    utils = sys.modules.setdefault(
        "whole_body_tracking.utils",
        types.ModuleType("whole_body_tracking.utils"),
    )
    package.utils = utils
    utils.training_contract = contract_mod
    sys.modules[
        "whole_body_tracking.utils.training_contract"
    ] = contract_mod
    if importlib.util.find_spec("hydra") is None:
        hydra = types.ModuleType("hydra")
        hydra.__spec__ = importlib.util.spec_from_loader(
            "hydra", loader=None
        )
        hydra.main = lambda *_args, **_kwargs: lambda function: function
        sys.modules["hydra"] = hydra
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.__spec__ = importlib.util.spec_from_loader(
            "omegaconf", loader=None
        )
        omegaconf.ListConfig = list
        omegaconf.OmegaConf = NS(
            resolve=lambda *_args, **_kwargs: None,
            set_struct=lambda *_args, **_kwargs: None,
        )
        sys.modules["omegaconf"] = omegaconf
    return _load_by_path(
        "_action_ball_bootstrap_train_under_test", TRAIN_PATH
    )


def _contract(contract_mod, *, action_count=1, ready_value=0.0):
    action_order = [f"action_{index}" for index in range(action_count)]
    joint_names = [f"joint_{index}" for index in range(31)]
    ready_q = [float(ready_value)] * 31
    default_q = [0.0] * 31
    scale = [1.0] * 31
    hard_lower = [-2.0] * 31
    hard_upper = [2.0] * 31
    return {
        "schema_version": 1,
        "kind": contract_mod.ACTION_BALL_POLICY_BOOTSTRAP_KIND,
        "action_count": action_count,
        "action_order": action_order,
        "joint_names": joint_names,
        "ready_source": {
            "semantics": (
                "motion.joint_pos[motion.seg_start[action_slot]]"
            ),
            "canonical_ready_sha256": "",
            "canonical_ready_fk_sha256": "",
            "motion_sha256_per_action": ["a" * 64] * action_count,
            "shared_ready_joint_pos": ready_q,
            "shared_ready_joint_pos_sha256": (
                contract_mod.action_ball_shared_ready_sha256(
                    action_order=action_order,
                    joint_names=joint_names,
                    shared_ready_joint_pos=ready_q,
                )
            ),
        },
        "decoder": {
            "semantics": "q_des=default_joint_pos+action_scale*action",
            "use_default_offset": True,
            "default_joint_pos": default_q,
            "action_scale": scale,
            "normalized_bias": ready_q,
            "startup_offset_delta_source": (
                "events.add_joint_default_pos.uniform_add"
            ),
            "startup_offset_delta_lower": [-0.01] * 31,
            "startup_offset_delta_upper": [0.01] * 31,
        },
        "initialization": {
            "fresh_only": True,
            "resume_overwrite_prohibited": True,
            "output_layer_weight": "zeros",
            "output_layer_bias": "decoder.normalized_bias",
            "init_noise_std": 0.02,
            "noise_std_type": "log",
            "required_realized_init_noise_std": 0.02,
            "sigma_envelope": 4.0,
        },
        "hard_inner_guard": {
            "limit_source": "articulation.data.joint_pos_limits",
            "margin_rad": 0.0,
            "margin_fraction": 0.02,
            "hard_lower": hard_lower,
            "hard_upper": hard_upper,
            "hard_inner_lower": [-1.92] * 31,
            "hard_inner_upper": [1.92] * 31,
        },
    }


def _agent_cfg():
    return NS(
        to_dict=lambda: {
            "num_steps_per_env": 24,
            "empirical_normalization": False,
            "policy": {
                "init_noise_std": 0.02,
                "noise_std_type": "log",
                "actor_hidden_dims": [64],
                "critic_hidden_dims": [64],
                "activation": "elu",
            },
            "algorithm": {
                "learning_rate": 0.0003,
                "rnd_cfg": None,
                "symmetry_cfg": None,
            },
        }
    )


def test_n1_and_n5_bootstrap_validate(contract_mod):
    for action_count in (1, 5):
        block = _contract(contract_mod, action_count=action_count)
        assert (
            contract_mod.validate_action_ball_policy_bootstrap(
                block, expected_action_count=action_count
            )["action_count"]
            == action_count
        )


def test_schema1_legacy_scalar_initialization_keys_remain_valid(
    contract_mod, train_mod
):
    torch = pytest.importorskip("torch")
    block = _contract(contract_mod, ready_value=0.25)
    block["initialization"].pop("noise_std_type")
    block["initialization"].pop("required_realized_init_noise_std")
    validated = contract_mod.validate_action_ball_policy_bootstrap(block)
    assert "noise_std_type" not in validated["initialization"]

    actor = torch.nn.Sequential(torch.nn.Linear(3, 31))
    runner = NS(
        alg=NS(
            policy=NS(
                actor=actor,
                noise_std_type="scalar",
                std=torch.full((31,), 0.02),
            )
        )
    )
    assert train_mod._apply_action_ball_fresh_policy_bootstrap(
        runner, block, checkpoint_path=None
    )
    assert torch.count_nonzero(actor[-1].weight).item() == 0
    assert torch.equal(actor[-1].bias, torch.full((31,), 0.25))


def test_n73_constant_bias_fails_closed(contract_mod):
    block = _contract(contract_mod, action_count=73)
    with pytest.raises(ValueError, match="supports exact N=1/N=5"):
        contract_mod.validate_action_ball_policy_bootstrap(block)


def test_four_sigma_hard_inner_gate_rejects_unsafe_ready(contract_mod):
    block = _contract(contract_mod, ready_value=1.90)
    with pytest.raises(ValueError, match="4-sigma"):
        contract_mod.validate_action_ball_policy_bootstrap(block)


def test_policy_recipe_hash_binds_full_bootstrap(
    contract_mod, train_mod
):
    block = _contract(contract_mod)
    recipe = train_mod._action_ball_agent_recipe(
        _agent_cfg(), policy_bootstrap=block
    )
    assert recipe["recipe"]["schema_version"] == 2
    assert recipe["recipe"]["policy_initialization"] == block
    changed = copy.deepcopy(block)
    changed["decoder"]["default_joint_pos"][0] = 0.01
    changed["decoder"]["normalized_bias"][0] = -0.01
    assert (
        train_mod._action_ball_agent_recipe(
            _agent_cfg(), policy_bootstrap=changed
        )["sha256"]
        != recipe["sha256"]
    )


def test_bootstrap_is_explicit_opt_in_and_legacy_recipe_stays_schema1(
    train_mod,
):
    requested, output = (
        train_mod._resolve_action_ball_shared_ready_bootstrap_request(
            {}, action_ball_launch_requested=True
        )
    )
    assert requested is False
    assert output is None
    recipe = train_mod._action_ball_agent_recipe(_agent_cfg())
    assert recipe["recipe"]["schema_version"] == 1
    assert "policy_initialization" not in recipe["recipe"]


def test_dynamic_ready_scalar_and_vendor_log_schema_paths_both_remain_reachable(
    train_mod,
):
    assert train_mod._action_ball_policy_bootstrap_schema_version(
        dynamic_ready=False, noise_std_type="scalar"
    ) == 1
    assert train_mod._action_ball_policy_bootstrap_schema_version(
        dynamic_ready=True, noise_std_type="scalar"
    ) == 2
    assert train_mod._action_ball_policy_bootstrap_schema_version(
        dynamic_ready=True, noise_std_type="log"
    ) == 3
    with pytest.raises(RuntimeError, match="scalar or log"):
        train_mod._action_ball_policy_bootstrap_schema_version(
            dynamic_ready=True, noise_std_type="unsupported"
        )


def test_policy_recipe_output_is_resolved_before_bootstrap_choice(
    train_mod, tmp_path,
):
    output_path = str(tmp_path / "recipe.json")
    requested, output = (
        train_mod._resolve_action_ball_shared_ready_bootstrap_request(
            {
                "action_ball_policy_recipe_output_path": output_path
            },
            action_ball_launch_requested=True,
        )
    )
    assert requested is False
    assert output == output_path


def test_policy_recipe_materialization_is_no_clobber_and_roundtrips(
    contract_mod, train_mod, tmp_path,
):
    bootstrap = _contract(contract_mod)
    recipe = train_mod._action_ball_agent_recipe(
        _agent_cfg(), policy_bootstrap=bootstrap
    )
    output = tmp_path / "recipe.json"
    document = train_mod._materialize_action_ball_policy_recipe(
        str(output),
        policy_recipe=recipe,
        policy_bootstrap=bootstrap,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert document["policy_contract_sha256"] == recipe["sha256"]
    with pytest.raises(RuntimeError, match="fresh no-clobber"):
        train_mod._materialize_action_ball_policy_recipe(
            str(output),
            policy_recipe=recipe,
            policy_bootstrap=bootstrap,
        )


def test_fresh_bootstrap_sets_last_layer_and_runtime_std(
    contract_mod, train_mod, capsys
):
    torch = pytest.importorskip("torch")
    block = _contract(contract_mod, ready_value=0.25)
    actor = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 31),
    )
    policy = NS(
        actor=actor,
        noise_std_type="log",
        log_std=torch.full((31,), math.log(0.02)),
    )
    runner = NS(alg=NS(policy=policy))
    assert train_mod._apply_action_ball_fresh_policy_bootstrap(
        runner, block, checkpoint_path=None
    )
    assert torch.count_nonzero(actor[-1].weight).item() == 0
    assert torch.equal(actor[-1].bias, torch.full((31,), 0.25))
    line = next(
        item
        for item in capsys.readouterr().out.splitlines()
        if item.startswith("HOPE_ACTION_BALL_POLICY_BOOTSTRAP_JSON=")
    )
    receipt = json.loads(line.split("=", 1)[1])
    assert receipt["noise_std_type"] == "log"
    assert receipt["parameter_name"] == "log_std"
    assert receipt["configured_init_noise_std"] == 0.02
    assert receipt["realized_policy_std_min"] == pytest.approx(
        0.02, rel=0.0, abs=1.0e-8
    )


def test_fresh_log_std_bootstrap_rejects_every_resume(
    contract_mod, train_mod
):
    torch = pytest.importorskip("torch")
    block = _contract(contract_mod, ready_value=0.25)
    actor = torch.nn.Sequential(torch.nn.Linear(3, 31))
    with torch.no_grad():
        actor[-1].weight.fill_(0.5)
        actor[-1].bias.fill_(-0.5)
    policy = NS(
        actor=actor,
        noise_std_type="log",
        log_std=torch.full((31,), math.log(0.02)),
    )
    runner = NS(alg=NS(policy=policy))
    before_weight = actor[-1].weight.detach().clone()
    before_bias = actor[-1].bias.detach().clone()
    with pytest.raises(RuntimeError, match="forbids every checkpoint"):
        train_mod._apply_action_ball_fresh_policy_bootstrap(
            runner, block, checkpoint_path="/exact/scalar_model_100.pt"
        )
    assert torch.equal(actor[-1].weight, before_weight)
    assert torch.equal(actor[-1].bias, before_bias)


def test_fresh_bootstrap_rejects_nonflat_std_parameter(
    contract_mod, train_mod
):
    torch = pytest.importorskip("torch")
    block = _contract(contract_mod, ready_value=0.25)
    actor = torch.nn.Sequential(torch.nn.Linear(3, 31))
    runner = NS(
        alg=NS(
            policy=NS(
                actor=actor,
                noise_std_type="log",
                log_std=torch.full((1, 31), math.log(0.02)),
            )
        )
    )
    with pytest.raises(RuntimeError, match="flat 31-action vector"):
        train_mod._apply_action_ball_fresh_policy_bootstrap(
            runner, block, checkpoint_path=None
        )


def test_legacy_scalar_resume_still_skips_without_overwrite(
    contract_mod, train_mod
):
    torch = pytest.importorskip("torch")
    block = _contract(contract_mod, ready_value=0.25)
    block["initialization"]["noise_std_type"] = "scalar"
    actor = torch.nn.Sequential(torch.nn.Linear(3, 31))
    with torch.no_grad():
        actor[-1].weight.fill_(0.5)
        actor[-1].bias.fill_(-0.5)
    runner = NS(
        alg=NS(
            policy=NS(
                actor=actor,
                noise_std_type="scalar",
                std=torch.full((31,), 0.02),
            )
        )
    )
    before_weight = actor[-1].weight.detach().clone()
    before_bias = actor[-1].bias.detach().clone()

    assert not train_mod._apply_action_ball_fresh_policy_bootstrap(
        runner, block, checkpoint_path="/exact/model_100.pt"
    )
    assert torch.equal(actor[-1].weight, before_weight)
    assert torch.equal(actor[-1].bias, before_bias)
