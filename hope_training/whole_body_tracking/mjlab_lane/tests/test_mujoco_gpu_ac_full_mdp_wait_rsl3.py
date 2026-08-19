from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys
import types

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]


def _load():
    path = LANE / "mujoco_gpu_ac_full_mdp_wait_rsl3.py"
    spec = importlib.util.spec_from_file_location("mujoco_wait_rsl3_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rsl3_config_keeps_fullmdp_actor_and_critic_groups_separate():
    module = _load()
    cfg = module.build_train_cfg()
    assert cfg["num_steps_per_env"] == 24
    assert cfg["obs_groups"] == {"policy": ["policy"], "critic": ["critic"]}
    assert cfg["policy"]["init_noise_std"] == 0.02
    assert cfg["policy"]["noise_std_type"] == "log"
    assert cfg["algorithm"]["num_learning_epochs"] == 5
    assert cfg["algorithm"]["num_mini_batches"] == 4
    assert _load().build_train_cfg(7)["num_steps_per_env"] == 7


@pytest.mark.parametrize(
    (
        "num_envs",
        "num_steps",
        "num_updates",
        "full_a_mode",
        "transitions",
        "lifecycle",
        "emit_evidence",
        "expected_evidence",
    ),
    (
        (2, 24, 1, False, 48, "idle_wait_only", False, None),
        (
            2,
            1,
            1,
            True,
            2,
            "full_a_slice_attempted",
            True,
            {
                "reveal_rows": 2,
                "launch_rows": 0,
                "flight_terminal_rows": 0,
                "selected_reset_rows": 0,
                "racket_contact_eligible_rows": 0,
                "racket_contact_rows": 0,
                "r03_present_rows": 2,
                "r03_physically_valid_rows": 2,
                "racket_contact_rate": None,
                "unmeasured": [],
            },
        ),
        (
            3,
            4,
            2,
            True,
            24,
            "full_a_slice_attempted",
            True,
            {
                "reveal_rows": 9,
                "launch_rows": 9,
                "flight_terminal_rows": 6,
                "selected_reset_rows": 6,
                "racket_contact_eligible_rows": 9,
                "racket_contact_rows": 3,
                "r03_present_rows": 24,
                "r03_physically_valid_rows": 24,
                "racket_contact_rate": 1.0 / 3.0,
                "unmeasured": [],
            },
        ),
        (
            2,
            1,
            1,
            True,
            2,
            "full_a_slice_attempted",
            False,
            {
                "reveal_rows": 0,
                "launch_rows": 0,
                "flight_terminal_rows": 0,
                "selected_reset_rows": 0,
                "racket_contact_eligible_rows": 0,
                "racket_contact_rows": 0,
                "r03_present_rows": 0,
                "r03_physically_valid_rows": 0,
                "racket_contact_rate": None,
                "unmeasured": [
                    "flight_terminal_rows",
                    "launch_rows",
                    "r03_physically_valid_rows",
                    "r03_present_rows",
                    "racket_contact_eligible_rows",
                    "racket_contact_rows",
                    "reveal_rows",
                    "selected_reset_rows",
                ],
            },
        ),
    ),
)
def test_main_orchestrates_one_update_without_logging_or_checkpoint(
    monkeypatch,
    capsys,
    tmp_path,
    num_envs,
    num_steps,
    num_updates,
    full_a_mode,
    transitions,
    lifecycle,
    emit_evidence,
    expected_evidence,
):
    module = _load()
    ready_pose = tmp_path / "ready_pose.json"
    payload = b'{"pose":"frozen"}'
    ready_pose.write_bytes(payload)
    monkeypatch.setenv("ACTIONBALL_READY_POSE", str(ready_pose))
    monkeypatch.setattr(module, "READY_POSE_SHA256", hashlib.sha256(payload).hexdigest())

    class _Cfg:
        def __init__(self, **values):
            vars(self).update(values)

    class _Env:
        def __init__(
            self,
            sim,
            task,
            device,
            seed,
            ready_pose_payload,
            ready_pose_source,
            full_a_mode,
        ):
            assert sim.nworld == num_envs and task.action_scale_mode == "vendor"
            assert device == "cuda:0" and seed == 0
            assert ready_pose_payload == payload and ready_pose_source == str(ready_pose)
            assert full_a_mode is expected_full_a_mode
            self.num_envs = num_envs
            self.num_actions = 31
            self.common_step_counter = 0
            self.full_a_mode = full_a_mode
            self.device = torch.device("cpu")
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self.max_episode_length = 150

        def get_observations(self):
            return {
                "policy": torch.zeros(num_envs, 229),
                "critic": torch.zeros(num_envs, 399),
            }

        def step(self, _actions):
            step_index = self.common_step_counter
            self.common_step_counter += 1
            extras = {}
            if self.full_a_mode and emit_evidence:
                phase = step_index % 3
                reveal = torch.full((num_envs,), phase == 0, dtype=torch.bool)
                launch = torch.full((num_envs,), phase == 1, dtype=torch.bool)
                terminal = torch.full((num_envs,), phase == 2, dtype=torch.bool)
                contact = torch.zeros(num_envs, dtype=torch.bool)
                if phase == 1:
                    contact[0] = True
                extras = {
                    "full_a_reveal_event": reveal,
                    "full_a_launch_event": launch,
                    "full_a_flight_terminal_event": terminal,
                    "full_a_selected_reset_event": terminal.clone(),
                    "full_a_racket_contact_eligible_event": launch.clone(),
                    "full_a_racket_contact_event": contact,
                    "full_a_r03_present_event": torch.ones_like(reveal),
                    "full_a_r03_physically_valid_event": torch.ones_like(reveal),
                }
            return (
                self.get_observations(),
                torch.ones(num_envs),
                torch.zeros(num_envs),
                extras,
            )

    wait_module = types.ModuleType("mujoco_gpu_ac_full_mdp_initial_wait_env")
    wait_module.__file__ = str(
        LANE / "mujoco_gpu_ac_full_mdp_initial_wait_env.py"
    )
    wait_module.FullMdpInitialWaitVecEnv = _Env
    wait_module.SimCfg = _Cfg
    wait_module.TaskCfg = _Cfg

    class _Algorithm:
        def __init__(self):
            self.optimizer = types.SimpleNamespace(state={"parameter": {}})
            self.storage = types.SimpleNamespace(step=0, rewards=torch.ones(1))

        def update(self):
            return {"surrogate": torch.tensor(0.0)}

    class _Runner:
        def __init__(self, env, cfg, log_dir, device):
            assert log_dir is None and device == "cuda:0"
            assert cfg["obs_groups"]["critic"] == ["critic"]
            assert cfg["num_steps_per_env"] == num_steps
            self.env = env
            self.alg = _Algorithm()
            self.disable_logs = False

        def learn(self, iterations, init_at_random_ep_len):
            assert iterations == num_updates and init_at_random_ep_len is False
            assert self.disable_logs is True
            for _ in range(iterations):
                for _ in range(num_steps):
                    self.env.step(torch.zeros(num_envs, 31))
                self.alg.update()

    monkeypatch.setitem(sys.modules, wait_module.__name__, wait_module)
    monkeypatch.setattr(module, "_rsl3_runner", lambda: ("3.1.2", _Runner, object()))
    monkeypatch.setattr(module, "_require_rsl3_runtime", lambda *_: None)

    expected_full_a_mode = full_a_mode
    assert (
        module.main(
            num_envs=num_envs,
            num_steps_per_env=num_steps,
            num_updates=num_updates,
            full_a_mode=full_a_mode,
        )
        == 0
    )
    line = capsys.readouterr().out.strip()
    prefix = "ACTION_BALL_MUJOCO_WAIT_RSL3_JSON="
    assert line.startswith(prefix)
    record = json.loads(line.removeprefix(prefix))
    assert record["ppo_update_calls"] == num_updates
    assert record["transitions"] == transitions
    assert record["task_lifecycle"] == lifecycle
    if full_a_mode:
        assert record["full_a_complete"] is False
        assert record["full_a_slice_evidence"] == expected_evidence
        assert record["not_produced"] == {
            "selected_rubber_contact": True,
            "r06_landing_outcome": True,
            "r07_recovery": True,
            "reward_terms_10_13": True,
        }
    else:
        assert "full_a_complete" not in record
        assert "full_a_slice_evidence" not in record
        assert "not_produced" not in record


def test_ready_pose_binding_rejects_missing_relative_symlink_and_wrong_bytes(
    monkeypatch, tmp_path
):
    module = _load()
    monkeypatch.delenv("ACTIONBALL_READY_POSE", raising=False)
    with pytest.raises(RuntimeError, match="not bound"):
        module._ready_pose_input()

    monkeypatch.setenv("ACTIONBALL_READY_POSE", "ready_pose.json")
    with pytest.raises(RuntimeError, match="path differs"):
        module._ready_pose_input()

    target = tmp_path / "ready_pose.json"
    target.write_text("{}", encoding="utf-8")
    alias = tmp_path / "ready_pose_alias.json"
    alias.symlink_to(target)
    monkeypatch.setenv("ACTIONBALL_READY_POSE", str(alias))
    with pytest.raises(RuntimeError, match="path differs"):
        module._ready_pose_input()

    monkeypatch.setenv("ACTIONBALL_READY_POSE", str(target))
    with pytest.raises(RuntimeError, match="path differs"):
        module._ready_pose_input()


def test_foreign_preloaded_wait_environment_is_rejected(monkeypatch):
    module = _load()
    foreign = types.ModuleType("mujoco_gpu_ac_full_mdp_initial_wait_env")
    foreign.__file__ = "/tmp/foreign/mujoco_gpu_ac_full_mdp_initial_wait_env.py"
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(RuntimeError, match="import origin differs"):
        module._wait_module()


def test_foreign_preloaded_rsl_runner_is_rejected(monkeypatch):
    module = _load()
    foreign = types.ModuleType("rsl_rl.runners.on_policy_runner")
    foreign.__file__ = "/tmp/foreign/rsl_rl/runners/on_policy_runner.py"

    class _ForeignRunner:
        pass

    _ForeignRunner.__module__ = foreign.__name__
    foreign.OnPolicyRunner = _ForeignRunner
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    distribution = types.SimpleNamespace(
        version="3.1.2",
        locate_file=lambda path: LANE / "overlay" / path,
    )
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda _: distribution)
    with pytest.raises(RuntimeError, match="RSL-RL import origin differs"):
        module._rsl3_runner()


def test_foreign_rsl_algorithm_is_rejected_after_runner_construction():
    module = _load()

    class _ForeignAlgorithm:
        pass

    runner = types.SimpleNamespace(alg=_ForeignAlgorithm())
    distribution = types.SimpleNamespace(
        locate_file=lambda path: LANE / "overlay" / path,
    )
    with pytest.raises(RuntimeError, match="runtime origin differs"):
        module._require_rsl3_runtime(distribution, runner, torch)


def test_foreign_ppo_binding_is_rejected_before_runner_construction():
    module = _load()
    source = str(Path(__file__).resolve())

    class _CanonicalPPO:
        pass

    class _ForeignPPO:
        pass

    class _ActorCritic:
        pass

    class _ActorCriticRecurrent:
        pass

    class _RolloutStorage:
        pass

    runner_module = types.SimpleNamespace(
        __file__=source,
        PPO=_ForeignPPO,
        ActorCritic=_ActorCritic,
        ActorCriticRecurrent=_ActorCriticRecurrent,
    )
    ppo_module = types.SimpleNamespace(
        __file__=source,
        PPO=_CanonicalPPO,
        RolloutStorage=_RolloutStorage,
        optim=types.SimpleNamespace(Adam=torch.optim.Adam),
    )
    actor_module = types.SimpleNamespace(__file__=source, ActorCritic=_ActorCritic)
    recurrent_module = types.SimpleNamespace(
        __file__=source, ActorCriticRecurrent=_ActorCriticRecurrent
    )
    storage_module = types.SimpleNamespace(
        __file__=source, RolloutStorage=_RolloutStorage
    )
    distribution = types.SimpleNamespace(locate_file=lambda _: source)
    with pytest.raises(RuntimeError, match="preconstruction origin differs"):
        module._require_rsl3_preconstruction(
            distribution,
            runner_module,
            ppo_module,
            actor_module,
            recurrent_module,
            storage_module,
            torch,
        )


@pytest.mark.skipif(
    os.environ.get("ACTIONBALL_RUN_MUJOCO_GPU_RSL3") != "1",
    reason="set ACTIONBALL_RUN_MUJOCO_GPU_RSL3=1 on the isolated RSL3 GPU stack",
)
def test_real_wait_environment_runs_one_real_rsl3_update(capsys):
    module = _load()
    assert module.main() == 0
    line = capsys.readouterr().out
    assert "ACTION_BALL_MUJOCO_WAIT_RSL3_JSON=" in line
    assert '"ppo_update_calls": 1' in line
    assert '"environment_steps": 24' in line
    assert '"transitions": 48' in line
    assert '"policy_width": 229' in line
    assert '"critic_width": 399' in line
