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
SOURCE_COMMIT = "a" * 40
RUN_NAMESPACE = "mujoco-full-a-runner-v2-test"
ACTION_CONTRACT = {
    "action_joint_order_contract_id": "a3-gmr-dof-pos-to-runtime-articulation-v1",
    "action_joint_order_contract_sha256": (
        "b09987ff7a1bfa624b566cc8884d16672ba73c1acc3f92efb8a4faa99d314815"
    ),
    "action_offset_source": "runtime_plant.default_joint_pos_rad",
    "action_offset_sha256": (
        "1b638d7b2e1ac7e552aace2ac8c2b00980dd9daf691f930b5fe775cebc84af78"
    ),
    "full_a_reset_joint_source": "runtime_plant.default_joint_pos_rad",
    "full_a_reset_root_source": "AGIBOT_A3_CFG.init_state.pos/rot",
    "full_a_policy_bootstrap": "a3_default_stand_zero_head_v1",
    "raw_action_clip": None,
    "executable_qdes_guard": "mujoco_hard_range_only_divergent_declared",
    "transfer_authority": False,
    "matched_cross_backend_authority": False,
}


def _load():
    path = LANE / "mujoco_gpu_ac_full_mdp_wait_rsl3.py"
    spec = importlib.util.spec_from_file_location("mujoco_wait_rsl3_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_consumer():
    path = LANE / "mujoco_full_mdp_longrun_consumer.py"
    spec = importlib.util.spec_from_file_location("mujoco_consumer_chain_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_a_policy_bootstrap_zeroes_only_output_head_and_pins_std():
    module = _load()
    actor = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.ELU(), torch.nn.Linear(8, 31)
    )
    hidden_before = actor[0].weight.detach().clone()
    policy = types.SimpleNamespace(
        actor=actor,
        log_std=torch.nn.Parameter(
            torch.full((31,), float(torch.log(torch.tensor(0.02))))
        ),
        noise_std_type="log",
    )
    runner = types.SimpleNamespace(alg=types.SimpleNamespace(policy=policy))

    module._apply_full_a_policy_bootstrap(runner, torch)

    assert torch.equal(actor[0].weight, hidden_before)
    assert torch.count_nonzero(actor[-1].weight) == 0
    assert torch.count_nonzero(actor[-1].bias) == 0
    torch.testing.assert_close(torch.exp(policy.log_std), torch.full((31,), 0.02))


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


FULL_A_EVENT_KEYS = (
    "full_a_reveal_event", "full_a_reveal_due_event",
    "full_a_reveal_deferred_event", "full_a_launch_event",
    "full_a_flight_terminal_event", "full_a_shot_retired_event",
    "full_a_completed_action_epoch_event",
    "full_a_selected_reset_event",
    "full_a_racket_contact_eligible_event", "full_a_racket_contact_event",
    "full_a_selected_contact_event", "full_a_opposite_contact_event",
    "full_a_edge_contact_event", "full_a_between_contact_event",
    "full_a_invalid_contact_event", "full_a_r03_present_event",
    "full_a_r03_physically_valid_event", "full_a_landing_crossing_event",
    "full_a_r06_present_event", "full_a_r06_eligible_event",
    "full_a_r06_common_event", "full_a_r07_present_event",
    "full_a_r07_eligible_event", "full_a_recovery_success_event",
    "full_a_recovery_failure_event", "full_a_recovery_timeout_event",
)


def _install_fake_stack(
    monkeypatch, tmp_path, *, num_envs, num_steps, num_updates,
    full_a_mode, schema_ok=True, fail_optimizer=False,
    drop_optimizer_state=False, empty_optimizer_state=False,
    valid_torch_snapshot=False,
):
    module = _load()
    ready_pose = tmp_path / "ready_pose.json"
    ready_payload = b'{"pose":"frozen"}'
    ready_pose.write_bytes(ready_payload)
    monkeypatch.setenv("ACTIONBALL_READY_POSE", str(ready_pose))
    monkeypatch.setattr(
        module, "READY_POSE_SHA256", hashlib.sha256(ready_payload).hexdigest()
    )
    trace, saved = [], []
    evidence = tmp_path / "updates.jsonl"
    snapshots = tmp_path / "snapshots"
    completion = tmp_path / "completion.json"
    snapshots.mkdir()

    ledger_path = LANE / "mujoco_full_mdp_update_ledger.py"
    ledger_name = "mujoco_full_mdp_update_ledger_runner_test"
    spec = importlib.util.spec_from_file_location(ledger_name, ledger_path)
    assert spec is not None and spec.loader is not None
    ledger_module = importlib.util.module_from_spec(spec)
    sys.modules[ledger_name] = ledger_module
    spec.loader.exec_module(ledger_module)

    class _TracingLedger(ledger_module.FullMdpUpdateLedger):
        def prepare(self, *args, **kwargs):
            trace.append("prepare")
            return super().prepare(*args, **kwargs)

        def ack(self, *args, **kwargs):
            payload = super().ack(*args, **kwargs)
            trace.append("ack")
            return payload

    monkeypatch.setattr(
        module, "_update_ledger_module",
        lambda: types.SimpleNamespace(FullMdpUpdateLedger=_TracingLedger),
    )

    class _Cfg:
        def __init__(self, **values):
            vars(self).update(values)

    class _Env:
        def __init__(
            self, sim, task, device, seed, ready_pose_payload,
            ready_pose_source, full_a_mode,
        ):
            assert sim.nworld == num_envs and task.action_scale_mode == "vendor"
            assert device == "cuda:0" and seed == 0
            assert ready_pose_payload == ready_payload
            assert ready_pose_source == str(ready_pose)
            assert full_a_mode is expected_mode
            assert task.episode_length_s == (30.0 if full_a_mode else 3.0)
            self.num_envs, self.num_actions = num_envs, 31
            self.common_step_counter = 0
            self.full_a_mode = full_a_mode
            self.device = torch.device("cpu")
            self.reset_generation = torch.zeros(num_envs, dtype=torch.long)
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self.max_episode_length = 150

        @property
        def action_contract_identity(self):
            return dict(ACTION_CONTRACT)

        def get_observations(self):
            return {
                "policy": torch.zeros(num_envs, 229),
                "critic": torch.zeros(num_envs, 399),
            }

        def step(self, _actions):
            self.common_step_counter += 1
            terms = torch.zeros(num_envs, 20)
            terms[:, 14] = 1.0
            extras = {}
            if self.full_a_mode:
                extras = {
                    key: torch.zeros(num_envs, dtype=torch.bool)
                    for key in FULL_A_EVENT_KEYS
                }
                extras.update({
                    "time_outs": torch.zeros(num_envs, dtype=torch.bool),
                    "termination_bits": torch.zeros(num_envs, dtype=torch.long),
                    "backend_resolved_table_contact": torch.zeros(
                        num_envs, dtype=torch.bool
                    ),
                    "reward_terms": terms,
                    "reset_generation": self.reset_generation.clone(),
                    "full_a_action_slot": torch.zeros(num_envs, dtype=torch.long),
                    "full_a_action_uid": torch.full(
                        (num_envs,), module.FULL_A_ACTION_UID, dtype=torch.long
                    ),
                    "full_a_mount_normal_sign": torch.ones(
                        num_envs, dtype=torch.int8
                    ),
                    "full_a_contact_classification_status": torch.zeros(
                        num_envs, dtype=torch.int8
                    ),
                    "full_a_outcome_code": torch.zeros(
                        num_envs, dtype=torch.long
                    ),
                    "full_a_phase_before_reset": torch.zeros(
                        num_envs, dtype=torch.long
                    ).fill_(2),
                    "full_a_landing_on_opponent": torch.zeros(
                        num_envs, dtype=torch.bool
                    ),
                    "full_a_landing_opponent_bound": torch.zeros(
                        num_envs, dtype=torch.bool
                    ),
                })
                if not schema_ok:
                    extras.pop("time_outs")
            return self.get_observations(), terms.sum(1), torch.zeros(
                num_envs, dtype=torch.long
            ), extras

    wait_module = types.ModuleType("mujoco_gpu_ac_full_mdp_initial_wait_env")
    wait_module.__file__ = str(LANE / "mujoco_gpu_ac_full_mdp_initial_wait_env.py")
    wait_module.FullMdpInitialWaitVecEnv = _Env
    wait_module.SimCfg = _Cfg
    wait_module.TaskCfg = _Cfg
    wait_module.FULLMDP_TERMINATION_BITS = {
        "time_out": 1, "base_fell_tilt": 2, "base_too_low": 4,
        "joint_qdes_forbidden": 8, "robot_hit_table": 16,
    }

    class _Algorithm:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.zeros(2, 3))
            state = {} if empty_optimizer_state else {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(self.parameter),
                "exp_avg_sq": torch.zeros_like(self.parameter),
            }
            self.optimizer = types.SimpleNamespace(
                state={self.parameter: state},
                param_groups=[{"params": [self.parameter]}],
            )
            shape = (num_steps, num_envs, 1)
            self.storage = types.SimpleNamespace(
                step=0, rewards=torch.ones(shape), returns=torch.ones(shape),
                advantages=torch.ones(shape),
            )
            self.learning_rate = 1.0e-3
            class _Policy:
                def __init__(self):
                    self.actor = torch.nn.Sequential(torch.nn.Linear(1, 31))
                    self.log_std = torch.nn.Parameter(
                        torch.full((31,), float(torch.log(torch.tensor(0.02))))
                    )
                    self.noise_std_type = "log"

                @property
                def action_std(self):
                    return torch.exp(self.log_std).expand(num_envs, 31)

            self.policy = _Policy()

        def update(self):
            trace.append("optimizer")
            if fail_optimizer:
                raise RuntimeError("optimizer failed")
            self.storage.step = 0
            if drop_optimizer_state:
                self.optimizer.state = {}
            return {"value_function": 0.25, "surrogate": -0.125, "entropy": 1.5}

    class _Runner:
        def __init__(self, env, cfg, log_dir, device):
            assert log_dir is None and device == "cuda:0"
            assert cfg["num_steps_per_env"] == num_steps
            self.env, self.alg = env, _Algorithm()
            self.disable_logs = False
            self.current_learning_iteration = 0

        def learn(self, iterations, init_at_random_ep_len):
            assert iterations == num_updates and init_at_random_ep_len is False
            assert self.disable_logs is True
            for index in range(iterations):
                for _ in range(num_steps):
                    self.env.step(torch.zeros(num_envs, 31))
                self.alg.storage.step = num_steps
                self.alg.update()
                self.current_learning_iteration = index

        def save(self, stream, infos=None):
            assert self.logger_type == "tensorboard" and self.disable_logs is True
            trace.append("save")
            saved.append((self.current_learning_iteration, dict(infos)))
            if not valid_torch_snapshot:
                stream.write(b"diagnostic snapshot")
                return
            consumer = _load_consumer()
            model = {
                name: torch.zeros(shape, dtype=torch.float32)
                for name, shape in consumer.MODEL_SHAPES
            }
            parameter_ids = list(range(len(consumer.MODEL_SHAPES)))
            optimizer = {
                "state": {
                    index: {
                        "step": torch.tensor(1.0),
                        "exp_avg": torch.zeros(shape),
                        "exp_avg_sq": torch.zeros(shape),
                    }
                    for index, (_name, shape) in enumerate(consumer.MODEL_SHAPES)
                },
                "param_groups": [{"params": parameter_ids, "lr": 1.0e-3}],
            }
            torch.save({
                "model_state_dict": model,
                "optimizer_state_dict": optimizer,
                "iter": self.current_learning_iteration,
                "infos": dict(infos),
            }, stream)

    monkeypatch.setitem(sys.modules, wait_module.__name__, wait_module)
    monkeypatch.setattr(module, "_rsl3_runner", lambda: ("3.1.2", _Runner, object()))
    monkeypatch.setattr(module, "_require_rsl3_runtime", lambda *_: None)
    expected_mode = full_a_mode

    def invoke():
        kwargs = dict(
            num_envs=num_envs, num_steps_per_env=num_steps,
            num_updates=num_updates, full_a_mode=full_a_mode,
        )
        if full_a_mode:
            kwargs.update(
                evidence_jsonl=str(evidence), snapshot_dir=str(snapshots),
                completion_json=str(completion), source_commit=SOURCE_COMMIT,
                run_namespace=RUN_NAMESPACE, _test_allow_small_full_a=True,
            )
        return module.main(**kwargs)

    return invoke, trace, saved, evidence, snapshots, completion


def test_real_runner_writer_prefix_is_consumed_without_a_second_schema(
    monkeypatch, tmp_path,
):
    invoke, _trace, _saved, evidence, snapshots, _completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_steps=2, num_updates=1,
        full_a_mode=True, valid_torch_snapshot=True,
    )
    assert invoke() == 0
    consumer = _load_consumer()
    monkeypatch.setattr(consumer, "NUM_ENVS", 2)
    monkeypatch.setattr(consumer, "STEPS_PER_UPDATE", 2)
    monkeypatch.setattr(consumer, "TRANSITIONS_PER_UPDATE", 4)
    summary = consumer.consume(
        evidence,
        expected_updates=1,
        expected_source_commit=SOURCE_COMMIT,
        expected_run_namespace=RUN_NAMESPACE,
        snapshot_dir=snapshots,
    )
    assert summary["evidence_level"] == "advisory_prefix"
    assert summary["engineering_run_complete"] is False
    assert summary["business_chain_complete"] is False
    assert summary["snapshot_count"] == 1


def test_main_preserves_default_wait_learn_one(monkeypatch, capsys, tmp_path):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_steps=24, num_updates=1,
        full_a_mode=False,
    )
    assert invoke() == 0
    record = json.loads(
        capsys.readouterr().out.split("ACTION_BALL_MUJOCO_WAIT_RSL3_JSON=", 1)[1]
    )
    assert record["ppo_update_calls"] == 1
    assert record["environment_steps"] == 24
    assert record["transitions"] == 48
    assert record["task_lifecycle"] == "idle_wait_only"
    assert trace == ["optimizer"] and saved == []
    assert not evidence.exists() and list(snapshots.iterdir()) == []
    assert not completion.exists()


def test_full_a_orders_prepare_optimizer_ack_snapshot_and_keeps_zero_telemetry(
    monkeypatch, capsys, tmp_path,
):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=3, num_steps=2, num_updates=2,
        full_a_mode=True,
    )
    assert invoke() == 0
    output = capsys.readouterr().out.splitlines()
    assert trace == [
        "prepare", "optimizer", "save", "ack",
        "prepare", "optimizer", "save", "ack",
    ]
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert [row["update_index"] for row in rows] == [0, 1]
    assert all(row["schema_version"] == 2 and row["run_identity"] == {
        "source_commit": SOURCE_COMMIT, "run_namespace": RUN_NAMESPACE,
    } for row in rows)
    assert rows[0]["extras_counts"]["r06_present_rows"] == 0
    assert rows[1]["extras_counts"]["r07_present_rows"] == 0
    assert rows[1]["reward20"]["reward20_finite_rows"] == 6
    assert rows[1]["optimizer_metrics"] == {
        "entropy": 1.5, "surrogate": -0.125, "value_function": 0.25
    }
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "model_0.pt", "model_1.pt"
    ]
    assert [row[0] for row in saved] == [0, 1]
    for index, (_saved_index, infos) in enumerate(saved):
        assert infos == {
            "diagnostic_unauthorized": True,
            "checkpoint_authority": False,
            "resume_authority": False,
            "update_index": index,
            "completed_updates": index + 1,
            "run_identity": {
                "source_commit": SOURCE_COMMIT, "run_namespace": RUN_NAMESPACE,
            },
            "prepared_update_sha256": rows[index]["prepared_update_sha256"],
        }
        assert rows[index]["snapshot"]["name"] == f"model_{index}.pt"
        assert rows[index]["snapshot"]["sha256"] == hashlib.sha256(
            (snapshots / f"model_{index}.pt").read_bytes()
        ).hexdigest()
    seal = json.loads(completion.read_text())
    assert seal["schema_version"] == 2
    assert seal["record_type"] == "mujoco_full_mdp_completion"
    assert seal["diagnostic_unauthorized"] is True
    assert seal["checkpoint_authority"] is False
    assert seal["resume_authority"] is False
    assert seal["run_identity"] == {
        "source_commit": SOURCE_COMMIT, "run_namespace": RUN_NAMESPACE,
    }
    assert seal["action_contract"] == ACTION_CONTRACT
    assert seal["evidence_jsonl"] == {
        "bytes": evidence.stat().st_size,
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    assert seal["snapshot_receipts"] == [row["snapshot"] for row in rows]
    assert all(seal[name] is True for name in (
        "final_observation_finite", "rollout_storage_finite",
        "optimizer_state_present", "optimizer_state_finite",
    ))
    final = json.loads(output[-1].split("=", 1)[1])
    assert final["full_a_update_ack_count"] == 2
    assert final["engineering_run_complete"] is True
    assert final["task_lifecycle"] == "full_a_engineering_longrun_complete"
    assert "full_a_complete" not in final
    assert "not_produced" not in final


@pytest.mark.parametrize(
    "schema_ok,fail_optimizer,error,expected_trace",
    (
        (False, False, "time_outs", ["prepare"]),
        (True, True, "optimizer failed", ["prepare", "optimizer"]),
    ),
)
def test_full_a_failure_has_no_ack_or_snapshot(
    monkeypatch, tmp_path, schema_ok, fail_optimizer, error, expected_trace,
):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_steps=1, num_updates=1,
        full_a_mode=True, schema_ok=schema_ok, fail_optimizer=fail_optimizer,
    )
    with pytest.raises(RuntimeError, match=error):
        invoke()
    assert trace == expected_trace
    assert evidence.read_bytes() == b""
    assert list(snapshots.iterdir()) == [] and saved == []
    assert not completion.exists()


@pytest.mark.parametrize("state_fault", ("drop", "empty"))
def test_full_a_final_gate_withholds_completion_after_durable_ack(
    monkeypatch, tmp_path, state_fault,
):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_steps=1, num_updates=1,
        full_a_mode=True, drop_optimizer_state=state_fault == "drop",
        empty_optimizer_state=state_fault == "empty",
    )
    with pytest.raises(RuntimeError, match="update evidence differs"):
        invoke()
    assert trace == ["prepare", "optimizer", "save", "ack"]
    assert len(evidence.read_text().splitlines()) == 1
    assert [path.name for path in snapshots.iterdir()] == ["model_0.pt"]
    assert saved and not completion.exists()


def test_full_a_production_shape_and_snapshot_schedule_are_exact(
    monkeypatch, tmp_path,
):
    module = _load()
    monkeypatch.setattr(
        module, "_rsl3_runner",
        lambda: pytest.fail("shape validation must precede RSL construction"),
    )
    with pytest.raises(ValueError, match="4096x24x25000"):
        module.main(
            num_envs=2, num_steps_per_env=24, num_updates=25_000,
            full_a_mode=True, evidence_jsonl=str(tmp_path / "updates.jsonl"),
            snapshot_dir=str(tmp_path), completion_json=str(tmp_path / "seal.json"),
            source_commit=SOURCE_COMMIT, run_namespace=RUN_NAMESPACE,
        )

    cadence_root = tmp_path / "cadence"
    cadence_root.mkdir()
    invoke, _trace, _saved, _evidence, snapshots, _completion = _install_fake_stack(
        monkeypatch, cadence_root, num_envs=1, num_steps=1,
        num_updates=1002, full_a_mode=True,
    )
    assert invoke() == 0
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "model_0.pt", "model_1000.pt", "model_1001.pt",
    ]


def test_evidence_jsonl_is_created_exclusively(monkeypatch, tmp_path):
    module = _load()
    existing = tmp_path / "updates.jsonl"
    existing.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._open_evidence_jsonl(str(existing))

    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "occupied.pt").write_bytes(b"occupied")
    with pytest.raises(ValueError, match="snapshot directory differs"):
        module._snapshot_root(str(snapshots))

    seal = tmp_path / "completion.json"
    seal.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        module._write_completion(str(seal), {"schema_version": 2})
    assert seal.read_bytes() == b"do not overwrite"


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

    class _MLP:
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
    actor_module = types.SimpleNamespace(
        __file__=source, ActorCritic=_ActorCritic, MLP=_MLP
    )
    recurrent_module = types.SimpleNamespace(
        __file__=source, ActorCriticRecurrent=_ActorCriticRecurrent
    )
    storage_module = types.SimpleNamespace(
        __file__=source, RolloutStorage=_RolloutStorage
    )
    mlp_module = types.SimpleNamespace(__file__=source, MLP=_MLP)
    distribution = types.SimpleNamespace(locate_file=lambda _: source)
    with pytest.raises(RuntimeError, match="preconstruction origin differs"):
        module._require_rsl3_preconstruction(
            distribution,
            runner_module,
            ppo_module,
            actor_module,
            recurrent_module,
            storage_module,
            mlp_module,
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
