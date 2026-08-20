from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import uuid

import torch


WBT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WBT_ROOT / "source" / "whole_body_tracking"
RUNNER_PATH = (
    SOURCE_ROOT
    / "whole_body_tracking"
    / "utils"
    / "my_on_policy_runner.py"
)

class _Memory:
    def __init__(self) -> None:
        self.hidden_states = None

    def reset(self, dones=None, hidden_states=None) -> None:
        if dones is None:
            self.hidden_states = hidden_states
        elif self.hidden_states is not None:
            self.hidden_states[..., dones == 1, :] = 0.0

class _Policy(torch.nn.Module):
    is_recurrent = True
    noise_std_type = "scalar"

    def __init__(self, num_envs: int, num_actions: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((num_actions,), 0.25))
        self.std = torch.nn.Parameter(torch.full((num_actions,), 0.20))
        self.memory_a = _Memory()
        self.memory_c = _Memory()
        self._num_envs = num_envs

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

class _Normalizer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.register_buffer("count", torch.zeros(()))
        self.forward_calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        self.count.add_(1.0)
        return (value - self.mean) / self.std

class _Transition:
    _FIELDS = (
        "observations",
        "privileged_observations",
        "actions",
        "privileged_actions",
        "rewards",
        "dones",
        "values",
        "actions_log_prob",
        "action_mean",
        "action_sigma",
        "hidden_states",
        "rnd_state",
    )

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        for name in self._FIELDS:
            setattr(self, name, None)

class _Storage:
    def __init__(self) -> None:
        self.step = 0

    def clear(self) -> None:
        self.step = 0

class _Algorithm:
    def __init__(self, policy: _Policy, *, leave_storage_full: bool) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-2)
        self.schedule = "adaptive"
        self.desired_kl = 0.01
        self.learning_rate = 1.0e-2
        self.rnd = None
        self.storage = _Storage()
        self.transition = _Transition()
        self.leave_storage_full = leave_storage_full
        self.first_act_observations = None
        self.first_act_critic_observations = None
        self.first_act_hidden = None

    def act(self, observations: torch.Tensor, critic: torch.Tensor) -> torch.Tensor:
        if self.first_act_observations is None:
            self.first_act_observations = observations.detach().clone()
            self.first_act_critic_observations = critic.detach().clone()
            hidden = self.policy.get_hidden_states()
            self.first_act_hidden = tuple(
                None if item is None else item.detach().clone() for item in hidden
            )
        actor_hidden, critic_hidden = self.policy.get_hidden_states()
        if actor_hidden is None:
            actor_hidden = torch.zeros(1, observations.shape[0], 3)
            critic_hidden = torch.zeros(1, observations.shape[0], 3)
        increment = observations[:, :1].reshape(1, observations.shape[0], 1)
        self.policy.memory_a.hidden_states = actor_hidden + increment
        self.policy.memory_c.hidden_states = critic_hidden + increment + 0.5
        self.transition.observations = observations
        self.transition.privileged_observations = observations
        self.transition.actions = torch.zeros(observations.shape[0], 2)
        self.transition.hidden_states = self.policy.get_hidden_states()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos) -> None:
        del rewards, infos
        self.storage.step += 1
        self.transition.clear()
        self.policy.memory_a.reset(dones)
        self.policy.memory_c.reset(dones)

    def compute_returns(self, _critic: torch.Tensor) -> None:
        return None

    def update(self):
        self.optimizer.zero_grad()
        loss = sum(parameter.square().sum() for parameter in self.policy.parameters())
        loss.backward()
        self.optimizer.step()
        self.learning_rate *= 0.5
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
        if not self.leave_storage_full:
            self.storage.clear()
        return {"loss": float(loss.detach())}

class _Env:
    def __init__(
        self,
        *,
        raw_offset: float = 0.0,
        obs_mode: str = "action_ball_a211",
        target_mode: str = "action_ball",
        actor_width: int = 211,
        critic_width: int = 319,
    ) -> None:
        self.unwrapped = self
        self.num_envs = 2
        self.num_actions = 2
        self.device = "cpu"
        self.getter_calls = 0
        self.noise_calls = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.raw_offset = float(raw_offset)
        self.actor_width = int(actor_width)
        self.critic_width = int(critic_width)
        self._action_ball_full_mdp_runtime_lease = object()
        self._drain_owner = None
        self.cfg = SimpleNamespace(
            obs_mode=obs_mode,
            commands=SimpleNamespace(
                racket_target=SimpleNamespace(target_mode=target_mode)
            ),
        )

    @property
    def action_ball_full_mdp_runtime_lease(self):
        return self._action_ball_full_mdp_runtime_lease

    def action_ball_full_mdp_ppo_drain_owner(self, lease):
        if lease is not self._action_ball_full_mdp_runtime_lease:
            raise RuntimeError("foreign test runtime lease")
        if self._drain_owner is None:
            raise RuntimeError("test global drain is not installed")
        return self._drain_owner

    def _observations(self, value: float):
        actor = torch.full((self.num_envs, self.actor_width), value)
        critic = torch.full((self.num_envs, self.critic_width), value + 0.25)
        actor[:, -1] = 1.0
        critic[:, -1] = 1.0
        return actor, {"observations": {"critic": critic}}

    def get_observations(self):
        self.getter_calls += 1
        self.noise_calls += 1
        return self._observations(self.raw_offset + self.step_calls)

    def reset(self):
        self.reset_calls += 1
        raise AssertionError("test R10 lane must not reset")

    def step(self, _actions):
        self.step_calls += 1
        actor, extras = self._observations(self.raw_offset + self.step_calls)
        rewards = torch.zeros(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        return actor, rewards, dones, extras

def _load_runner_module():
    saved_modules: dict[str, object] = {}

    def install(name: str, module: types.ModuleType) -> None:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    rsl_rl = types.ModuleType("rsl_rl")
    rsl_env = types.ModuleType("rsl_rl.env")
    rsl_runners = types.ModuleType("rsl_rl.runners")
    rsl_on_policy = types.ModuleType("rsl_rl.runners.on_policy_runner")

    class VecEnv:
        pass

    class OnPolicyRunner:
        def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
            self.env = env
            self.cfg = train_cfg
            self.log_dir = log_dir
            self.device = device
            self.training_type = "rl"
            observations, extras = env.get_observations()
            assert "critic" in extras["observations"]
            self.privileged_obs_type = "critic"
            self.num_steps_per_env = int(train_cfg["num_steps_per_env"])
            self.save_interval = 100
            self.empirical_normalization = True
            self.obs_normalizer = _Normalizer(observations.shape[1])
            self.privileged_obs_normalizer = _Normalizer(
                extras["observations"]["critic"].shape[1]
            )
            policy = _Policy(env.num_envs, env.num_actions)
            self.alg = _Algorithm(
                policy,
                leave_storage_full=bool(train_cfg.get("leave_storage_full", False)),
            )
            self.disable_logs = True
            self.is_distributed = False
            self.current_learning_iteration = 0
            self.tot_timesteps = 0
            self.tot_time = 0.0

        def train_mode(self):
            self.alg.policy.train()
            self.obs_normalizer.train()
            self.privileged_obs_normalizer.train()

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            assert init_at_random_ep_len is False
            observations, extras = self.env.get_observations()
            critic = extras["observations"]["critic"]
            observations = observations.to(self.device)
            critic = critic.to(self.device)
            self.train_mode()
            start = int(self.current_learning_iteration)
            for iteration in range(start, start + int(num_learning_iterations)):
                with torch.inference_mode():
                    for _ in range(self.num_steps_per_env):
                        actions = self.alg.act(observations, critic)
                        observations, rewards, dones, infos = self.env.step(actions)
                        observations = self.obs_normalizer(observations.to(self.device))
                        critic = self.privileged_obs_normalizer(
                            infos["observations"]["critic"].to(self.device)
                        )
                        self.alg.process_env_step(rewards, dones, infos)
                    self.alg.compute_returns(critic)
                self.alg.update()
                self.current_learning_iteration = iteration

    OnPolicyRunner.__module__ = "rsl_rl.runners.on_policy_runner"
    rsl_env.VecEnv = VecEnv
    rsl_on_policy.OnPolicyRunner = OnPolicyRunner
    install("rsl_rl", rsl_rl)
    install("rsl_rl.env", rsl_env)
    install("rsl_rl.runners", rsl_runners)
    install("rsl_rl.runners.on_policy_runner", rsl_on_policy)

    isaaclab_rl = types.ModuleType("isaaclab_rl")
    isaaclab_rsl = types.ModuleType("isaaclab_rl.rsl_rl")
    isaaclab_rsl.export_policy_as_onnx = lambda *_args, **_kwargs: None
    install("isaaclab_rl", isaaclab_rl)
    install("isaaclab_rl.rsl_rl", isaaclab_rsl)

    exporter = types.ModuleType("whole_body_tracking.utils.exporter")
    exporter.attach_onnx_metadata = lambda *_args, **_kwargs: None
    exporter.export_motion_policy_as_onnx = lambda *_args, **_kwargs: False
    exporter.is_empirical_normalizer = lambda value: isinstance(value, _Normalizer)
    install("whole_body_tracking.utils.exporter", exporter)

    contract = types.ModuleType("whole_body_tracking.utils.training_contract")
    contract.CHECKPOINT_CONTRACT_LINEAGE_EXACT_KEY = "lineage"
    contract.CHECKPOINT_CONTRACT_SCHEMA_KEY = "schema"
    contract.CHECKPOINT_CONTRACT_SHA_KEY = "sha"
    contract.CHECKPOINT_LAUNCH_CLAIM_SHA_KEY = "claim"
    contract.TRAINING_CONTRACT_SCHEMA_VERSION = 1
    contract.validate_training_launch_claim_sha256 = lambda value: value
    install("whole_body_tracking.utils.training_contract", contract)

    for family in ("a211", "c211"):
        name = (
            "whole_body_tracking.tasks.tracking."
            f"action_ball_{family}_trainability"
        )
        module = types.ModuleType(name)
        function_name = (
            "validate_action_ball_211_runner"
            if family == "a211"
            else "validate_action_ball_c211_runner"
        )
        setattr(
            module,
            function_name,
            lambda _runner, _family=family: {"family": _family},
        )
        install(name, module)

    module_name = f"_r10_runner_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    install(module_name, module)
    spec.loader.exec_module(module)
    cls = module.MotionOnPolicyRunner
    module._test_original_validate_task_first_exact_resume_terms = (
        cls._validate_task_first_exact_resume_terms
    )
    cls._validate_task_first_exact_resume_terms = lambda _self: None
    cls._emit_rsl_rl_runtime_abi = lambda _self, **_kwargs: None
    cls._emit_control_step_action_delay_runtime_receipt = lambda _self: None
    cls._reward_ppo_economy_gate_requested = lambda _self: False
    cls._effective_reward_activation_task_kind = lambda _self: None
    cls._diagnostic_joint_safety_compact_evidence = lambda _self: False
    cls._bind_joint_safety_action_term = lambda _self, **_kwargs: None
    cls._emit_policy_std_update = lambda _self, **_kwargs: None
    cls._notify_command_terms_rollout_end = lambda _self, _step: None
    cls._service_action_ball_frozen_evaluation = lambda _self, _step: False
    return module, saved_modules
