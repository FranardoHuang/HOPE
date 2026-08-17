"""Independent tensor/config tests for the fresh full-MDP policy birth prior."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"
TRAIN_PATH = SCRIPTS / "train.py"
COMMON_TASK_PATH = (
    ROOT / "cfg" / "task" / "HOPEPingPongActionBallFullMdpCommon.yaml"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

_TEMP_IMPORT_STUBS = []
if "hydra" not in sys.modules and importlib.util.find_spec("hydra") is None:
    hydra_stub = types.ModuleType("hydra")
    hydra_stub.main = lambda **_kwargs: lambda function: function
    sys.modules["hydra"] = hydra_stub
    _TEMP_IMPORT_STUBS.append("hydra")
if "omegaconf" not in sys.modules and importlib.util.find_spec("omegaconf") is None:
    omega_stub = types.ModuleType("omegaconf")

    class _ListConfig(list):
        pass

    class _OmegaConf:
        pass

    omega_stub.ListConfig = _ListConfig
    omega_stub.OmegaConf = _OmegaConf
    sys.modules["omegaconf"] = omega_stub
    _TEMP_IMPORT_STUBS.append("omegaconf")

import train as train_mod  # noqa: E402

for _stub_name in _TEMP_IMPORT_STUBS:
    sys.modules.pop(_stub_name, None)


class _Manager:
    def __init__(self, term):
        self._term = term

    def get_term(self, name):
        assert name == "joint_pos"
        return self._term


class _AgentCfg:
    def __init__(self, *, noise=0.02, noise_std_type="log"):
        self._value = {
            "policy": {
                "init_noise_std": noise,
                "noise_std_type": noise_std_type,
            }
        }

    def to_dict(self):
        return {
            "policy": dict(self._value["policy"]),
        }


def _config(**overrides):
    value = {
        "kind": "a3_default_stand_zero_head_v1",
        "init_noise_std": 0.02,
        "noise_std_type": "log",
    }
    value.update(overrides)
    return value


def _env(*, offset_delta=0.0, use_default_offset=True):
    default = torch.linspace(-0.3, 0.3, 31).repeat(2, 1)
    # Row 1 differs so a scalar/nominal-only decoder cannot pass this fixture.
    default[1] += 0.01
    offset = default.clone()
    offset[1, 7] += offset_delta
    scale = torch.linspace(0.05, 0.25, 31).repeat(2, 1)
    action = types.SimpleNamespace(
        _joint_ids=slice(None),
        _offset=offset,
        _scale=scale,
        cfg=types.SimpleNamespace(use_default_offset=use_default_offset),
    )
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            default_joint_pos=default,
            joint_names=[f"joint_{index}" for index in range(31)],
        )
    )

    class _NoMotionAuthority:
        def __getattr__(self, name):
            raise AssertionError(f"bootstrap queried forbidden Motion authority: {name}")

    return types.SimpleNamespace(
        action_manager=_Manager(action),
        scene={"robot": robot},
        command_manager=_NoMotionAuthority(),
    )


def _contract(**env_kwargs):
    return train_mod._action_ball_full_mdp_policy_bootstrap_contract(
        _env(**env_kwargs), _AgentCfg(), _config()
    )


def _runner(*, std=0.02):
    actor = torch.nn.Sequential(
        torch.nn.Linear(6, 8),
        torch.nn.ELU(),
        torch.nn.Linear(8, 31),
    )
    policy = types.SimpleNamespace(
        actor=actor,
        noise_std_type="log",
        log_std=torch.nn.Parameter(torch.log(torch.full((31,), std))),
    )
    return types.SimpleNamespace(
        alg=types.SimpleNamespace(policy=policy)
    )


def _pre_gym_binding(run_mode):
    return train_mod._ActionBallFullMdpPreGymBinding(
        owner_type=type("Owner", (), {}),
        owner_factory=object(),
        dependency_dag_sha256=None,
        dependency_kind=None,
        epoch_owner_type=None,
        gym_entry_point="entry",
        run_mode=run_mode,
        launch_authorized=run_mode == train_mod._ACTION_BALL_FULL_MDP_FORMAL_MODE,
        diagnostic_unauthorized=(
            run_mode
            == train_mod._ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ),
    )


def _joint_safety_cfg(*, compact=False):
    return types.SimpleNamespace(
        actions=types.SimpleNamespace(
            joint_pos=types.SimpleNamespace(
                pre_apply_guard_diagnostic_compact_evidence=compact
            )
        )
    )


def test_common_ac_config_pins_one_exact_full_mdp_bootstrap():
    source = COMMON_TASK_PATH.read_text(encoding="utf-8")
    assert source.count("action_ball_full_mdp_policy_bootstrap:") == 1
    assert source.count("kind: a3_default_stand_zero_head_v1") == 1
    assert source.count("init_noise_std: 0.02") == 1
    assert source.count("noise_std_type: log") == 1


def test_diagnostic_binding_alone_selects_consumed_compact_joint_safety():
    cfg = _joint_safety_cfg()
    mode = train_mod._configure_action_ball_full_mdp_joint_safety_evidence(
        cfg,
        _pre_gym_binding(
            train_mod._ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
        ),
    )
    assert mode == "diagnostic_compact_two_phase_update_v1"
    assert cfg.actions.joint_pos.pre_apply_guard_diagnostic_compact_evidence is True

    with pytest.raises(RuntimeError, match="not by a caller/task override"):
        train_mod._configure_action_ball_full_mdp_joint_safety_evidence(
            _joint_safety_cfg(compact=True),
            _pre_gym_binding(
                train_mod._ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE
            ),
        )


def test_formal_binding_keeps_dense_joint_safety_and_rejects_compact_override():
    cfg = _joint_safety_cfg()
    mode = train_mod._configure_action_ball_full_mdp_joint_safety_evidence(
        cfg,
        _pre_gym_binding(train_mod._ACTION_BALL_FULL_MDP_FORMAL_MODE),
    )
    assert mode == "formal_policy_step_summary_v1"
    assert cfg.actions.joint_pos.pre_apply_guard_diagnostic_compact_evidence is False

    with pytest.raises(RuntimeError, match="formal full-MDP forbids"):
        train_mod._configure_action_ball_full_mdp_joint_safety_evidence(
            _joint_safety_cfg(compact=True),
            _pre_gym_binding(train_mod._ACTION_BALL_FULL_MDP_FORMAL_MODE),
        )


@pytest.mark.parametrize(
    "gym_id",
    (
        "HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0",
        "HOPE-PingPong-ActionBall-FullMdpC-AgibotA3-v0",
    ),
)
def test_registered_ac_parse_then_launch_binding_is_the_only_compact_writer(gym_id):
    pytest.importorskip("gymnasium")
    pytest.importorskip("isaaclab")
    pytest.importorskip("isaaclab_tasks")
    import whole_body_tracking  # noqa: F401
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    diagnostic_cfg = parse_env_cfg(gym_id, device="cpu", num_envs=2)
    diagnostic_action = diagnostic_cfg.actions.joint_pos
    assert diagnostic_action.pre_apply_guard_diagnostic_compact_evidence is False
    diagnostic_mode = train_mod._configure_action_ball_full_mdp_joint_safety_evidence(
        diagnostic_cfg,
        _pre_gym_binding(train_mod._ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE),
    )
    assert diagnostic_mode == "diagnostic_compact_two_phase_update_v1"
    assert diagnostic_action.pre_apply_guard_diagnostic_compact_evidence is True

    formal_cfg = parse_env_cfg(gym_id, device="cpu", num_envs=2)
    formal_action = formal_cfg.actions.joint_pos
    assert formal_action.pre_apply_guard_diagnostic_compact_evidence is False
    formal_mode = train_mod._configure_action_ball_full_mdp_joint_safety_evidence(
        formal_cfg,
        _pre_gym_binding(train_mod._ACTION_BALL_FULL_MDP_FORMAL_MODE),
    )
    assert formal_mode == "formal_policy_step_summary_v1"
    assert formal_action.pre_apply_guard_diagnostic_compact_evidence is False


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ({"kind": "motion_frame0"}, "kind must be exactly"),
        ({"init_noise_std": 1.0}, "must be exactly 0.02"),
        ({"noise_std_type": "scalar"}, "must be exactly 'log'"),
        ({"extra": True}, "silently ignored"),
    ],
)
def test_config_rejects_every_nonreviewed_bootstrap(fault, message):
    task = {
        "action_ball_full_mdp_runtime": True,
        "action_ball_full_mdp_policy_bootstrap": _config(**fault),
    }
    with pytest.raises(train_mod._OverrideError, match=message):
        train_mod._resolve_action_ball_full_mdp_policy_bootstrap_config(
            task, requested=True
        )


def test_legacy_task_cannot_consume_the_full_mdp_bootstrap_block():
    with pytest.raises(train_mod._OverrideError, match="valid only"):
        train_mod._resolve_action_ball_full_mdp_policy_bootstrap_config(
            {"action_ball_full_mdp_policy_bootstrap": _config()},
            requested=False,
        )


def test_algo_override_is_exact_and_happens_before_runner_cfg_parsing():
    algo = {
        "policy": {"init_noise_std": 1.0, "noise_std_type": "scalar"}
    }
    train_mod._apply_action_ball_full_mdp_policy_algo_config(algo, _config())
    assert algo["policy"] == {
        "init_noise_std": 0.02,
        "noise_std_type": "log",
    }
    run_source = inspect.getsource(train_mod._run_with_environment_close_owner)
    assert run_source.index(
        "_apply_action_ball_full_mdp_policy_algo_config("
    ) < run_source.index("agent_cfg = RslRlOnPolicyRunnerCfg(")


def test_contract_uses_live_per_row_default_offset_and_never_motion():
    contract = _contract()
    assert contract["kind"] == "a3_default_stand_zero_head_v1"
    assert contract["actor_output_dim"] == 31
    assert contract["initialization"]["output_layer_weight"] == "zeros"
    assert contract["initialization"]["output_layer_bias"] == [0.0] * 31
    assert contract["decoder"]["zero_action_target"] == (
        "robot.data.default_joint_pos[:,action_joint_ids]"
    )
    assert not any("sha" in key for key in contract)
    assert "motion" not in repr(contract).lower()


def test_live_action_term_counterexamples_fail_independently():
    with pytest.raises(RuntimeError, match="offset differs"):
        _contract(offset_delta=1.0e-4)
    with pytest.raises(RuntimeError, match="use_default_offset=True"):
        _contract(use_default_offset=False)
    with pytest.raises(RuntimeError, match="runner policy differs"):
        train_mod._action_ball_full_mdp_policy_bootstrap_contract(
            _env(), _AgentCfg(noise=1.0, noise_std_type="scalar"), _config()
        )


def test_fresh_apply_zeros_real_actor_for_distinct_observations(capsys):
    runner = _runner()
    contract = _contract()
    assert train_mod._apply_action_ball_full_mdp_fresh_policy_bootstrap(
        runner, contract, checkpoint_path=None
    )
    actor = runner.alg.policy.actor
    observations = torch.stack(
        (torch.zeros(6), torch.tensor([2.0, -3.0, 5.0, 7.0, -11.0, 13.0]))
    )
    assert torch.equal(actor(observations), torch.zeros(2, 31))
    assert torch.equal(
        torch.exp(runner.alg.policy.log_std), torch.full((31,), 0.02)
    )
    marker = capsys.readouterr().out
    assert marker.count("HOPE_ACTION_BALL_FULL_MDP_POLICY_BOOTSTRAP_JSON=") == 1
    assert '"actor_output_weight_nonzero_count":0' in marker
    assert '"actor_output_bias_nonzero_count":0' in marker


def test_actor_and_std_mutations_are_observed_from_real_tensors():
    runner = _runner()
    contract = _contract()
    train_mod._apply_action_ball_full_mdp_fresh_policy_bootstrap(
        runner, contract, checkpoint_path=None
    )
    with torch.no_grad():
        runner.alg.policy.actor[-1].weight[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="differs from zero-head"):
        train_mod._inspect_action_ball_full_mdp_policy_bootstrap_runtime(
            runner, contract, require_initial_values=True
        )
    runner.alg.policy.log_std = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(RuntimeError, match=r"log_std\[31\]"):
        train_mod._inspect_action_ball_full_mdp_policy_bootstrap_runtime(
            runner, contract, require_initial_values=False
        )


def test_resume_apply_and_postload_validation_never_mutate_learned_policy(capsys):
    runner = _runner(std=0.07)
    contract = _contract()
    with torch.no_grad():
        runner.alg.policy.actor[-1].weight.fill_(0.125)
        runner.alg.policy.actor[-1].bias.fill_(-0.25)
    before = {
        name: tensor.detach().clone()
        for name, tensor in runner.alg.policy.actor.state_dict().items()
    }
    before_log_std = runner.alg.policy.log_std.detach().clone()
    assert not train_mod._apply_action_ball_full_mdp_fresh_policy_bootstrap(
        runner, contract, checkpoint_path="/checkpoint/model_200.pt"
    )
    facts = train_mod._validate_action_ball_full_mdp_resumed_policy_bootstrap(
        runner, contract
    )
    assert facts["actor_output_weight_nonzero_count"] == 31 * 8
    assert facts["actor_output_bias_nonzero_count"] == 31
    for name, tensor in runner.alg.policy.actor.state_dict().items():
        assert torch.equal(tensor, before[name])
    assert torch.equal(runner.alg.policy.log_std, before_log_std)
    output = capsys.readouterr().out
    assert "HOPE_ACTION_BALL_FULL_MDP_POLICY_BOOTSTRAP_JSON=" not in output
    assert output.count("HOPE_ACTION_BALL_FULL_MDP_POLICY_RESUME_JSON=") == 1


def test_persisted_full_mdp_contract_contains_the_exact_bootstrap():
    class _Owner:
        pass

    binding = train_mod._ActionBallFullMdpPreGymBinding(
        owner_type=_Owner,
        owner_factory=object(),
        dependency_dag_sha256=None,
        dependency_kind="action_ball_epoch_runtime_dependencies_v1",
        epoch_owner_type=object,
        gym_entry_point=train_mod._ACTION_BALL_FULL_MDP_GYM_ENTRY_POINT,
        run_mode=train_mod._ACTION_BALL_FULL_MDP_SINGLE_ACTION_LEAN_MODE,
        launch_authorized=False,
        diagnostic_unauthorized=True,
    )
    bootstrap = _contract()
    persisted = train_mod._action_ball_full_mdp_training_contract(
        binding, _Owner(), None, None, bootstrap
    )
    assert persisted["policy_bootstrap"] == bootstrap
    assert persisted["policy_bootstrap"] is not bootstrap
    assert not any("sha" in key for key in persisted["policy_bootstrap"])


def test_run_applies_fresh_before_load_and_validates_resume_after_load():
    source = inspect.getsource(train_mod._run_with_environment_close_owner)
    fresh_index = source.index("full_mdp_bootstrap_applied =")
    load_index = source.rindex("_load_requested_checkpoint()")
    resume_index = source.index(
        "_validate_action_ball_full_mdp_resumed_policy_bootstrap("
    )
    assert fresh_index < load_index < resume_index
