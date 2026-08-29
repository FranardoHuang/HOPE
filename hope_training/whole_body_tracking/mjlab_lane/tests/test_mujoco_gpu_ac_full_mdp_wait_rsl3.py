from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest
import torch


LANE = Path(__file__).resolve().parents[1]
PLANT_XML = LANE.parents[2] / (
    "agi/A3_MuJoCo_Sim/aimrt_mujoco_sim/src/models/bin/cfg/model/"
    "a3p_pingpong_0807/a3p_pingpong_0807.xml"
)
SOURCE_COMMIT = "a" * 40
RUN_NAMESPACE = "mujoco-full-a-runner-v2-test"
VERIFICATION_RECEIPT_SHA256 = "c" * 64
OWNER_LOCAL_FRAME_SHA256 = "d" * 64
MUJOCO_WARP_RUNTIME = {
    "schema_version": 1,
    "distribution": "mujoco-warp",
    "fork_id": "hope_mujoco_warp_epa48_v1",
    "version": "3.10.0.3+hope.epa48.1",
    "epa_horizon": 48,
    "types_py_sha256": (
        "391e421eeede84389d6c7daeae39b19ce43132d29c11f7f3c328a50011c7a696"
    ),
    "wheel_sha256": (
        "58f47b1c3b4249d82666f25d3a302ff5a215043a3d7a3b9445a5ca7ef15b561a"
    ),
    "build_receipt_sha256": (
        "336f6454296d3c062e26fb0c330d6dbca4b2fd0ad4e50f386f8a647db013e041"
    ),
    "import_scope": "fresh_run_local_site",
}
RSL_RL_RUNTIME = {
    "distribution": "rsl-rl-lib",
    "version": "3.1.2",
    "wheel_sha256": (
        "406867356b70920e99ed8fd12c5b3463a64895407cc3ed96c917fddb9bfae06d"
    ),
    "import_scope": "fresh_run_local_site",
}
MJLAB_RUNTIME = {
    "schema_version": 1,
    "distribution": "mjlab",
    "version": "1.5.3",
    "import_scope": "verified_venv_distribution",
    "selected_tree_scope": "mjlab/**/*.py+mjlab/scene/scene.xml",
    "selected_file_count": 193,
    "selected_byte_count": 1_399_177,
    "selected_tree_sha256": (
        "88c9725d0416b4ac3e21f6752ad423c13ea3b8cfb9e23ca664f8aba146cec33d"
    ),
    "mjlab_tasks_entry_point_count": 0,
}


def _runtime_stack():
    return {
        "schema_version": 1,
        "mujoco_warp": dict(MUJOCO_WARP_RUNTIME),
        "rsl_rl": dict(RSL_RL_RUNTIME),
        "mjlab": dict(MJLAB_RUNTIME),
    }


def _plant_contract():
    name = "_runner_test_mujoco_full_mdp_plant_contract"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = LANE / "mujoco_full_mdp_plant_contract.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _plant_model():
    final_mjb = _plant_contract().expected_plant_model_identity()[
        "runtime_attach"
    ]["final_augmented_mjb"]
    return _plant_contract().verified_plant_model_identity(
        verification_receipt_sha256=VERIFICATION_RECEIPT_SHA256,
        owner_local_frame_sha256=OWNER_LOCAL_FRAME_SHA256,
        final_augmented_mjb=final_mjb,
    )


def _augmented_mjb():
    return dict(
        _plant_contract().expected_plant_model_identity()["runtime_attach"][
            "final_augmented_mjb"
        ]
    )


def _source_scan():
    source = _plant_contract().expected_plant_model_identity()["source_plant"]
    return {
        "root_path": str(PLANT_XML),
        **{key: source[key] for key in (
            "root_filename", "root_mjcf_sha256", "source_closure_sha256",
            "source_member_count", "source_total_bytes",
        )},
    }


def _identity():
    return {
        "source_commit": SOURCE_COMMIT,
        "run_namespace": RUN_NAMESPACE,
        "runtime_stack": _runtime_stack(),
        "plant_model": _plant_model(),
    }


ACTION_CONTRACT = {
    "action_joint_order_contract_id": "a3-gmr-dof-pos-to-runtime-articulation-v1",
    "action_joint_order_contract_sha256": (
        "b09987ff7a1bfa624b566cc8884d16672ba73c1acc3f92efb8a4faa99d314815"
    ),
    "action_offset_source": "runtime_plant.default_joint_pos_rad",
    "action_offset_sha256": (
        "1b638d7b2e1ac7e552aace2ac8c2b00980dd9daf691f930b5fe775cebc84af78"
    ),
    "full_a_reset_joint_source": "dynamic_ready.physical_ready.joint_pos_rad",
    "full_a_reset_root_source": "dynamic_ready.physical_ready.root_pose",
    "full_a_policy_bootstrap": "a3_take061_dynamic_ready_head_v1",
    "raw_action_clip": None,
    "executable_qdes_guard": "action_ball_shared_max_inward_state_guard_v2",
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
    source = _plant_contract().expected_plant_model_identity()["source_plant"]
    class Verified:
        portable_identity_sha256 = source["portable_identity_sha256"]
        verification_receipt_sha256 = VERIFICATION_RECEIPT_SHA256

        def consume_verified_model(self, consumer):
            return consumer(object())

    module._canonical_mujoco_identity_module = lambda: types.SimpleNamespace(
        verify_exact_mujoco_identity=lambda **_kwargs: Verified()
    )
    module._mujoco_module = lambda: object()
    module._table_termination_module = lambda: types.SimpleNamespace(
        consume_verified_owner_frame_contract=lambda _mujoco, verified: (
            verified.consume_verified_model(
                lambda _model: {"content_sha256": OWNER_LOCAL_FRAME_SHA256}
            )
        )
    )
    runtime_verification = object()
    module._epa48_runtime_module = lambda: types.SimpleNamespace(
        verify_runtime_stack_preimport=lambda: runtime_verification,
        verified_runtime_stack_identity=lambda actual: (
            _runtime_stack()
            if actual is runtime_verification
            else pytest.fail("consumer runtime verification token differs")
        ),
    )
    module._verified_runtime_mjb = lambda _evidence: _augmented_mjb()
    return module


def test_full_a_policy_bootstrap_sets_ready_mean_and_pins_std():
    module = _load()
    actor = torch.nn.Sequential(
        torch.nn.Linear(3, 8), torch.nn.ELU(), torch.nn.Linear(8, 31)
    )
    hidden_before = actor[0].weight.detach().clone()
    policy = types.SimpleNamespace(
        actor=actor,
        log_std=torch.nn.Parameter(
            torch.full(
                (31,),
                float(
                    torch.log(
                        torch.tensor(module.FULL_MDP_PPO_RECIPE.init_noise_std)
                    )
                ),
            )
        ),
        noise_std_type=module.FULL_MDP_PPO_RECIPE.noise_std_type,
    )
    runner = types.SimpleNamespace(alg=types.SimpleNamespace(policy=policy))
    ready_action = torch.linspace(-0.2, 0.4, 31)
    env = types.SimpleNamespace(_full_a_policy_bootstrap_action=ready_action)

    module._apply_full_a_policy_bootstrap(runner, torch, env)

    assert torch.equal(actor[0].weight, hidden_before)
    assert torch.count_nonzero(actor[-1].weight) == 0
    assert torch.equal(actor[-1].bias, ready_action)
    torch.testing.assert_close(
        torch.exp(policy.log_std),
        torch.full((31,), module.FULL_MDP_PPO_RECIPE.init_noise_std),
    )


def test_rsl3_config_keeps_fullmdp_actor_and_critic_groups_separate():
    module = _load()
    cfg = module.build_train_cfg()
    assert cfg["num_steps_per_env"] == 48
    assert cfg["save_interval"] == 2_000
    assert cfg["obs_groups"] == {"policy": ["policy"], "critic": ["critic"]}
    assert cfg["policy"]["init_noise_std"] == (
        module.FULL_MDP_PPO_RECIPE.init_noise_std
    )
    assert cfg["policy"]["noise_std_type"] == (
        module.FULL_MDP_PPO_RECIPE.noise_std_type
    )
    assert cfg["policy"]["actor_obs_normalization"] is False
    assert cfg["policy"]["critic_obs_normalization"] is False
    assert cfg["algorithm"]["num_learning_epochs"] == 5
    assert cfg["algorithm"]["num_mini_batches"] == 1
    assert cfg["algorithm"]["learning_rate"] == 1.0e-4
    assert cfg["algorithm"]["schedule"] == "fixed"
    assert cfg["algorithm"]["desired_kl"] is None
    assert cfg["algorithm"]["gamma"] == 0.99
    assert cfg["algorithm"]["lam"] == 0.98
    assert cfg["algorithm"]["entropy_coef"] == 0.0
    with pytest.raises(TypeError):
        module.build_train_cfg(7)
    with pytest.raises(TypeError):
        module.main(num_steps_per_env=7)


def test_full_a_binds_epa48_before_torch_rsl_and_wait_imports(tmp_path):
    trace = tmp_path / "import-order.trace"
    fake_stack = tmp_path / "fake_stack"
    fake_stack.mkdir()
    (fake_stack / "torch.py").write_text(
        "import os\n"
        "with open(os.environ['ACTIONBALL_IMPORT_TRACE'], 'a', encoding='utf-8') "
        "as stream:\n"
        "    stream.write('torch\\n')\n",
        encoding="utf-8",
    )
    runner = LANE / "mujoco_gpu_ac_full_mdp_wait_rsl3.py"
    script = textwrap.dedent(
        """
        import importlib.util
        from pathlib import Path
        import sys
        import types

        runner, fake_stack, trace, root = map(Path, sys.argv[1:])
        sys.path.insert(0, str(fake_stack))
        spec = importlib.util.spec_from_file_location("behavioral_runner", runner)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runtime_stack = {
            "schema_version": 1,
            "mujoco_warp": {"verified": "epa48"},
            "rsl_rl": {"verified": "rsl3"},
            "mjlab": {"verified": "mjlab-1.5.3"},
        }

        def log(value):
            with trace.open("a", encoding="utf-8") as stream:
                stream.write(value + "\\n")

        preimport = object()

        def verify_preimport():
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in ("torch", "mujoco_warp", "mjlab", "rsl_rl")
            )
            log("preverify")
            return preimport

        module._epa48_runtime_module = lambda: types.SimpleNamespace(
            verify_runtime_stack_preimport=verify_preimport,
        )

        def bind(raw, verified):
            assert raw == str(root / "runtime_site")
            assert verified is preimport
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in ("torch", "mujoco_warp", "mjlab", "rsl_rl")
            )
            log("bind")
            return runtime_stack

        def rsl3_runner():
            log("rsl")
            return "3.1.2", object(), object()

        def wait_module():
            log("wait")
            raise RuntimeError("stop-after-wait-import")

        module._bind_full_a_runtime = bind
        module._rsl3_runner = rsl3_runner
        module._wait_module = wait_module
        try:
            module.main(
                num_envs=1,
                num_updates=1,
                full_a_mode=True,
                evidence_jsonl=str(root / "updates.jsonl"),
                snapshot_dir=str(root / "snapshots"),
                completion_json=str(root / "completion.json"),
                source_commit="a" * 40,
                run_namespace="mujoco-full-a-import-order-test",
                mujoco_warp_runtime_site=str(root / "runtime_site"),
                _test_allow_small_full_a=True,
            )
        except RuntimeError as exc:
            assert str(exc) == "stop-after-wait-import"
        else:
            raise AssertionError("wait sentinel was not reached")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(runner), str(fake_stack), str(trace),
         str(tmp_path)],
        env={**os.environ, "ACTIONBALL_IMPORT_TRACE": str(trace)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "preverify", "bind", "torch", "rsl", "wait",
    ]


def test_full_a_run_identity_isolates_the_binder_result_copy():
    module = _load()
    runtime = _runtime_stack()
    plant_model = _plant_model()
    identity = module._run_identity(
        SOURCE_COMMIT, RUN_NAMESPACE, runtime, plant_model)
    assert identity == _identity()
    runtime["mujoco_warp"]["epa_horizon"] = 24
    plant_model["runtime_attach"]["final_augmented_mjb"]["sha256"] = "0" * 64
    assert identity == _identity()


def test_mjlab_postimport_verification_precedes_run_identity(monkeypatch):
    module = _load()
    runtime_stack = _runtime_stack()
    monkeypatch.setattr(
        module,
        "_epa48_runtime_module",
        lambda: types.SimpleNamespace(
            verify_loaded_mjlab_runtime_modules=lambda: dict(MJLAB_RUNTIME)
        ),
    )
    module._verify_full_a_runtime_postimport(runtime_stack)
    drifted = dict(MJLAB_RUNTIME)
    drifted["selected_tree_sha256"] = "0" * 64
    monkeypatch.setattr(
        module,
        "_epa48_runtime_module",
        lambda: types.SimpleNamespace(
            verify_loaded_mjlab_runtime_modules=lambda: drifted
        ),
    )
    with pytest.raises(RuntimeError, match="pre/post identity differs"):
        module._verify_full_a_runtime_postimport(runtime_stack)

    source = inspect.getsource(module.main)
    assert source.index("FullMdpInitialWaitVecEnv(") < source.index(
        "_verify_full_a_runtime_postimport(runtime_identity)"
    ) < source.index("_run_identity(")


def test_source_commit_is_measured_from_the_clean_runner_checkout(tmp_path):
    module = _load()
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Runner Test"],
        check=True,
    )
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()

    assert module._verified_source_checkout_commit(
        commit, repo_root=repo,
    ) == commit
    with pytest.raises(RuntimeError, match="source checkout differs"):
        module._verified_source_checkout_commit("a" * 40, repo_root=repo)

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source checkout differs"):
        module._verified_source_checkout_commit(commit, repo_root=repo)


def test_runtime_site_argument_is_full_a_only_and_required_before_torch(tmp_path):
    module = _load()
    with pytest.raises(ValueError, match="artifact arguments require --full-a"):
        module.main(mujoco_warp_runtime_site=str(tmp_path / "site"))
    with pytest.raises(ValueError, match="runtime site is not bound"):
        module.main(
            num_envs=1,
            num_updates=1,
            full_a_mode=True,
            evidence_jsonl=str(tmp_path / "updates.jsonl"),
            snapshot_dir=str(tmp_path),
            completion_json=str(tmp_path / "completion.json"),
            source_commit=SOURCE_COMMIT,
            run_namespace=RUN_NAMESPACE,
            _test_allow_small_full_a=True,
        )


FULL_A_EVENT_KEYS = (
    "full_a_scheduled_due_event", "full_a_due_terminal_overlap_event",
    "full_a_reveal_event", "full_a_reveal_due_event",
    "full_a_reveal_deferred_event", "full_a_launch_event",
    "full_a_missed_launch_event",
    "full_a_flight_terminal_event", "full_a_shot_retired_event",
    "full_a_completed_action_epoch_event",
    "full_a_selected_reset_event", "full_a_racket_contact_event",
    "full_a_selected_contact_event", "full_a_opposite_contact_event",
    "full_a_edge_contact_event", "full_a_between_contact_event",
    "full_a_invalid_contact_event", "full_a_actual_hard_edge_event",
    "full_a_qdes_guard_intervention_event",
    "full_a_r03_present_event",
    "full_a_r03_physically_valid_event", "full_a_landing_crossing_event",
    "full_a_r06_present_event", "full_a_r06_eligible_event",
    "full_a_r06_common_event", "full_a_r07_present_event",
    "full_a_r07_eligible_event", "full_a_recovery_success_event",
    "full_a_recovery_failure_event", "full_a_recovery_timeout_event",
    "full_a_recovery_completion_fault_event",
)


def _install_fake_stack(
    monkeypatch, tmp_path, *, num_envs, num_updates,
    full_a_mode, schema_ok=True, fail_optimizer=False,
    drop_optimizer_state=False, empty_optimizer_state=False,
    valid_torch_snapshot=False, diagnostic_rate_probe=False,
    storage_fault=None,
):
    module = _load()
    num_steps = module.NUM_STEPS_PER_ENV
    ready_pose = tmp_path / "ready_pose.json"
    ready_payload = b'{"pose":"frozen"}'
    ready_pose.write_bytes(ready_payload)
    monkeypatch.setenv("ACTIONBALL_READY_POSE", str(ready_pose))
    monkeypatch.setenv("A3_PINGPONG_XML", str(PLANT_XML))
    monkeypatch.setattr(
        module, "READY_POSE_SHA256", hashlib.sha256(ready_payload).hexdigest()
    )
    trace, saved, live_models = [], [], []
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
        lambda: types.SimpleNamespace(
            FullMdpUpdateLedger=_TracingLedger,
            STORAGE_FLOAT_WIDTHS=ledger_module.STORAGE_FLOAT_WIDTHS,
            storage_schema_is_exact=ledger_module.storage_schema_is_exact,
            storage_domain_validity=ledger_module.storage_domain_validity,
        ),
    )

    class _Cfg:
        def __init__(self, **values):
            vars(self).update(values)

    class _Env:
        def __init__(
            self, sim, task, device, xml_path, seed, ready_pose_payload,
            ready_pose_source, full_a_mode,
        ):
            assert sim.nworld == num_envs and task.action_scale_mode == "vendor"
            assert device == "cuda:0" and seed == 0
            assert ready_pose_payload == ready_payload
            assert ready_pose_source == str(ready_pose)
            assert xml_path == (PLANT_XML if full_a_mode else None)
            assert full_a_mode is expected_mode
            assert task.episode_length_s == (30.0 if full_a_mode else 3.0)
            self.num_envs, self.num_actions = num_envs, 31
            self.common_step_counter = 0
            self.full_a_mode = full_a_mode
            self.device = torch.device("cpu")
            self._full_a_policy_bootstrap_action = torch.linspace(
                -0.2, 0.4, 31
            )
            self.reset_generation = torch.zeros(num_envs, dtype=torch.long)
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
            self.max_episode_length = 150
            self.env = types.SimpleNamespace(xml_path=PLANT_XML)
            source = _plant_contract().expected_plant_model_identity()[
                "source_plant"
            ]
            self._table_keepout = types.SimpleNamespace(
                plant_identity_receipt={
                    "root_mjcf_sha256": source["root_mjcf_sha256"],
                    "identity_manifest_sha256": source["manifest_sha256"],
                    "portable_identity_sha256": source[
                        "portable_identity_sha256"
                    ],
                    "verification_receipt_sha256": VERIFICATION_RECEIPT_SHA256,
                    "owner_local_frame_sha256": OWNER_LOCAL_FRAME_SHA256,
                }
            )
            self.decimation = 20
            self.step_dt = 0.02
            self.njmax_alloc = 572
            self.naconmax_alloc = 128 * num_envs
            self.mj_model = types.SimpleNamespace(
                opt=types.SimpleNamespace(
                    timestep=0.001,
                    integrator=0,
                    solver=2,
                    ccd_iterations=35,
                    noslip_iterations=0,
                )
            )
            live_models.append(self.mj_model)

        @property
        def action_contract_identity(self):
            return dict(ACTION_CONTRACT)

        def get_observations(self):
            policy_width = 215 if self.full_a_mode else 229
            critic_width = 231 if self.full_a_mode else 399
            return {
                "policy": torch.zeros(num_envs, policy_width),
                "critic": torch.zeros(num_envs, critic_width),
            }

        def step(self, _actions):
            self.common_step_counter += 1
            terms = torch.zeros(num_envs, ledger_module.REWARD_TERM_COUNT)
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
                    "full_a_paddle_prior_playback": torch.zeros(
                        num_envs, dtype=torch.bool
                    ),
                    "full_a_paddle_prior_error": torch.zeros(
                        (num_envs, 4), dtype=torch.float32
                    ),
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
                    "full_a_fact_integrity_fault_bits": torch.zeros(
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
    wait_module.observation_contract = types.SimpleNamespace(
        ACTOR_WIDTH_V1=229,
        CRITIC_WIDTH_V1=399,
        ACTOR_WIDTH_V3=215,
        CRITIC_WIDTH_V3=231,
    )
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
                step=0,
                observations={
                    "policy": torch.zeros(num_steps, num_envs, 215),
                    "critic": torch.zeros(num_steps, num_envs, 231),
                },
                actions=torch.zeros(num_steps, num_envs, 31),
                dones=torch.zeros(shape, dtype=torch.uint8),
                values=torch.zeros(shape),
                actions_log_prob=torch.zeros(shape),
                mu=torch.zeros(num_steps, num_envs, 31),
                sigma=torch.ones(num_steps, num_envs, 31),
                rewards=torch.ones(shape),
                returns=torch.ones(shape),
                advantages=torch.ones(shape),
            )
            if storage_fault in ("observations_policy", "observations_critic"):
                self.storage.observations[
                    storage_fault[len("observations_"):]
                ][0, 0, 0] = torch.nan
            elif storage_fault == "dones_binary":
                self.storage.dones[0, 0, 0] = 2
            elif storage_fault == "sigma_zero":
                self.storage.sigma[0, 0, 0] = 0.0
            elif storage_fault == "sigma_negative":
                self.storage.sigma[0, 0, 0] = -1.0
            elif storage_fault is not None:
                getattr(self.storage, storage_fault)[0, 0, 0] = torch.nan
            self.learning_rate = 1.0e-3
            class _Policy:
                def __init__(self):
                    self.actor = torch.nn.Sequential(torch.nn.Linear(1, 31))
                    self.log_std = torch.nn.Parameter(
                        torch.full(
                            (31,),
                            float(
                                torch.log(
                                    torch.tensor(
                                        module.FULL_MDP_PPO_RECIPE.init_noise_std
                                    )
                                )
                            ),
                        )
                    )
                    self.noise_std_type = (
                        module.FULL_MDP_PPO_RECIPE.noise_std_type
                    )

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
    monkeypatch.setattr(module, "_scan_plant_source", lambda _path: _source_scan())
    contract = module._plant_contract_module()
    expected_attach = contract.expected_plant_model_identity()["runtime_attach"]
    monkeypatch.setattr(module, "_mujoco_module", lambda: object())
    monkeypatch.setattr(
        module,
        "_geometry_source_identity",
        lambda: expected_attach["geometry_source_sha256"],
    )
    def persist_runtime_mjb(_mujoco, _model, root):
        assert len(live_models) == 1 and _model is live_models[0]
        target = Path(root) / "runtime.mjb"
        target.write_bytes(b"fake runner-owned augmented MJB")
        return dict(expected_attach["final_augmented_mjb"])

    monkeypatch.setattr(
        contract, "persist_augmented_runtime_mjb", persist_runtime_mjb,
    )
    runtime_site = tmp_path / "runtime_site"
    runtime_preimport = object()
    monkeypatch.setattr(
        module,
        "_epa48_runtime_module",
        lambda: types.SimpleNamespace(
            verify_runtime_stack_preimport=lambda: runtime_preimport,
            verify_loaded_mjlab_runtime_modules=lambda: dict(MJLAB_RUNTIME),
        ),
    )
    monkeypatch.setattr(
        module,
        "_bind_full_a_runtime",
        lambda raw, verified: (
            _runtime_stack()
            if raw == str(runtime_site) and verified is runtime_preimport
            else pytest.fail("Full-A runtime site argument differs")
        ),
    )
    expected_mode = full_a_mode

    def invoke():
        kwargs = dict(
            num_envs=num_envs, num_updates=num_updates,
            full_a_mode=full_a_mode,
        )
        if full_a_mode:
            kwargs.update(
                evidence_jsonl=str(evidence), source_commit=SOURCE_COMMIT,
                run_namespace=RUN_NAMESPACE, _test_allow_small_full_a=True,
                mujoco_warp_runtime_site=str(runtime_site),
                diagnostic_rate_probe=diagnostic_rate_probe,
            )
            if not diagnostic_rate_probe:
                kwargs.update(
                    snapshot_dir=str(snapshots), completion_json=str(completion)
                )
        return module.main(**kwargs)

    return invoke, trace, saved, evidence, snapshots, completion


def test_real_runner_writer_prefix_is_consumed_without_a_second_schema(
    monkeypatch, tmp_path,
):
    invoke, _trace, _saved, evidence, snapshots, _completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_updates=1,
        full_a_mode=True, valid_torch_snapshot=True,
    )
    assert invoke() == 0
    consumer = _load_consumer()
    monkeypatch.setattr(consumer, "NUM_ENVS", 2)
    monkeypatch.setattr(consumer, "STEPS_PER_UPDATE", 48)
    monkeypatch.setattr(consumer, "TRANSITIONS_PER_UPDATE", 96)
    summary = consumer.consume(
        evidence,
        expected_updates=1,
        expected_source_commit=SOURCE_COMMIT,
        expected_run_namespace=RUN_NAMESPACE,
        expected_plant_xml=PLANT_XML,
        snapshot_dir=snapshots,
    )
    assert summary["evidence_level"] == "advisory_prefix"
    assert summary["engineering_run_complete"] is False
    assert summary["producer_attested_milestone_coverage_complete"] is False
    assert summary["same_epoch_chain_replay_status"] == "not_produced"
    assert summary["snapshot_count"] == 1


def test_real_runner_completion_v5_round_trips_through_summary_v6(
    monkeypatch, tmp_path,
):
    invoke, _trace, _saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_updates=2,
        full_a_mode=True, valid_torch_snapshot=True,
    )
    assert invoke() == 0
    consumer = _load_consumer()
    monkeypatch.setattr(consumer, "NUM_ENVS", 2)
    monkeypatch.setattr(consumer, "STEPS_PER_UPDATE", 48)
    monkeypatch.setattr(consumer, "TRANSITIONS_PER_UPDATE", 96)
    monkeypatch.setattr(consumer, "COMPLETE_UPDATES", 2)

    summary = consumer.consume(
        evidence,
        expected_updates=2,
        expected_source_commit=SOURCE_COMMIT,
        expected_run_namespace=RUN_NAMESPACE,
        expected_plant_xml=PLANT_XML,
        snapshot_dir=snapshots,
        completion_json=completion,
    )

    assert json.loads(completion.read_text())["schema_version"] == 5
    assert summary["schema_version"] == 6
    assert summary["engineering_run_complete"] is True
    assert summary["completion_seal_verified"] is True
    assert summary["producer_attested_milestone_coverage_complete"] is False
    assert summary["same_epoch_chain_replay_status"] == "not_produced"


def test_stdout_marker_failure_is_best_effort(monkeypatch):
    module = _load()

    class BrokenStdout:
        def write(self, _payload):
            raise BrokenPipeError("closed pipe")

        def flush(self):
            raise AssertionError("flush must not follow a failed write")

    warnings = []
    monkeypatch.setattr(module.sys, "stdout", BrokenStdout())
    monkeypatch.setattr(
        module.sys,
        "stderr",
        types.SimpleNamespace(
            write=lambda payload: warnings.append(payload), flush=lambda: None
        ),
    )
    module._best_effort_stdout_marker("committed")
    assert len(warnings) == 1
    assert json.loads(warnings[0]) == {
        "error_type": "BrokenPipeError",
        "event": "action_ball_stdout_marker_failed",
    }


def test_stdout_and_warning_sink_failures_cannot_revoke_commit(monkeypatch):
    module = _load()

    class FlushFailure:
        def write(self, payload):
            return len(payload)

        def flush(self):
            raise OSError("flush failed")

    class WarningFailure:
        def write(self, _payload):
            raise OSError("stderr failed")

        def flush(self):
            raise OSError("stderr flush failed")

    monkeypatch.setattr(module.sys, "stdout", FlushFailure())
    monkeypatch.setattr(module.sys, "stderr", WarningFailure())
    module._best_effort_stdout_marker("committed")


def test_main_preserves_default_wait_learn_one(monkeypatch, capsys, tmp_path):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_updates=1,
        full_a_mode=False,
    )
    assert invoke() == 0
    record = json.loads(
        capsys.readouterr().out.split("ACTION_BALL_MUJOCO_WAIT_RSL3_JSON=", 1)[1]
    )
    assert record["ppo_update_calls"] == 1
    assert record["environment_steps"] == 48
    assert record["transitions"] == 96
    assert record["action_ball_full_mdp_ppo_recipe_sha256"] == (
        _load().FULL_MDP_PPO_RECIPE_SHA256
    )
    assert record["task_lifecycle"] == "idle_wait_only"
    assert record["policy_width"] == 229
    assert record["critic_width"] == 399
    assert trace == ["optimizer"] and saved == []
    assert not evidence.exists() and list(snapshots.iterdir()) == []
    assert not completion.exists()
    assert not (tmp_path / "runtime.mjb").exists()


def test_full_a_orders_prepare_optimizer_ack_snapshot_and_keeps_zero_telemetry(
    monkeypatch, capsys, tmp_path,
):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=3, num_updates=2,
        full_a_mode=True,
    )
    assert invoke() == 0
    output = capsys.readouterr().out.splitlines()
    final = json.loads(
        next(
            line.split("ACTION_BALL_MUJOCO_WAIT_RSL3_JSON=", 1)[1]
            for line in output
            if line.startswith("ACTION_BALL_MUJOCO_WAIT_RSL3_JSON=")
        )
    )
    assert final["policy_width"] == 215
    assert final["critic_width"] == 231
    assert trace == [
        "prepare", "optimizer", "save", "ack",
        "prepare", "optimizer", "save", "ack",
    ]
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert [row["update_index"] for row in rows] == [0, 1]
    assert all(
        row["schema_version"] == 10 and row["run_identity"] == _identity()
        for row in rows
    )
    expected_mjb = _augmented_mjb()
    assert expected_mjb["relative_locator"] == "runtime.mjb"
    assert all(
        row["run_identity"]["plant_model"]["runtime_attach"][
            "final_augmented_mjb"
        ] == expected_mjb
        for row in rows
    )
    assert (tmp_path / expected_mjb["relative_locator"]).read_bytes() == (
        b"fake runner-owned augmented MJB"
    )
    assert all(set(row["storage_finite"]) == {
        "observations_policy", "observations_critic", "actions", "values",
        "actions_log_prob", "mu", "sigma", "rewards", "returns",
        "advantages",
    } for row in rows)
    assert all(row["storage_domains"] == {
        "dones_binary": True, "sigma_positive": True,
    } for row in rows)
    assert rows[0]["extras_counts"]["r06_present_rows"] == 0
    assert rows[1]["extras_counts"]["r07_present_rows"] == 0
    assert rows[1]["reward_graph"]["reward_terms_finite_rows"] == 144
    assert rows[1]["reward_graph"]["term_count"] == len(
        rows[1]["reward_graph"]["term_names"]
    )
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
            "run_identity": _identity(),
            "action_ball_full_mdp_ppo_recipe_sha256": (
                _load().FULL_MDP_PPO_RECIPE_SHA256
            ),
            "prepared_update_sha256": rows[index]["prepared_update_sha256"],
        }
        assert rows[index]["snapshot"]["name"] == f"model_{index}.pt"
        assert rows[index]["snapshot"]["sha256"] == hashlib.sha256(
            (snapshots / f"model_{index}.pt").read_bytes()
        ).hexdigest()
    seal = json.loads(completion.read_text())
    assert seal["schema_version"] == 5
    assert seal["record_type"] == "mujoco_full_mdp_completion"
    assert seal["diagnostic_unauthorized"] is True
    assert seal["checkpoint_authority"] is False
    assert seal["resume_authority"] is False
    assert seal["run_identity"] == _identity()
    assert seal["run_identity"]["plant_model"]["runtime_attach"][
        "final_augmented_mjb"
    ] == expected_mjb
    assert seal["action_contract"] == ACTION_CONTRACT
    assert seal["action_ball_full_mdp_ppo_recipe_sha256"] == (
        _load().FULL_MDP_PPO_RECIPE_SHA256
    )
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
        monkeypatch, tmp_path, num_envs=2, num_updates=1,
        full_a_mode=True, schema_ok=schema_ok, fail_optimizer=fail_optimizer,
    )
    with pytest.raises(RuntimeError, match=error):
        invoke()
    assert trace == expected_trace
    assert evidence.read_bytes() == b""
    assert list(snapshots.iterdir()) == [] and saved == []
    assert not completion.exists()


@pytest.mark.parametrize(
    "storage_fault,error",
    (
        ("observations_policy", "storage is nonfinite"),
        ("observations_critic", "storage is nonfinite"),
        ("actions", "storage is nonfinite"),
        ("values", "storage is nonfinite"),
        ("actions_log_prob", "storage is nonfinite"),
        ("mu", "storage is nonfinite"),
        ("sigma", "storage is nonfinite"),
        ("rewards", "storage is nonfinite"),
        ("returns", "storage is nonfinite"),
        ("advantages", "storage is nonfinite"),
        ("dones_binary", "storage domain differs"),
        ("sigma_zero", "sigma_positive"),
        ("sigma_negative", "sigma_positive"),
    ),
)
def test_full_a_complete_storage_faults_fail_before_optimizer_or_ack(
    monkeypatch, tmp_path, storage_fault, error,
):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_updates=1,
        full_a_mode=True, storage_fault=storage_fault,
    )
    with pytest.raises(RuntimeError, match=error):
        invoke()
    assert trace == ["prepare"]
    assert evidence.read_bytes() == b""
    assert list(snapshots.iterdir()) == [] and saved == []
    assert not completion.exists()


@pytest.mark.parametrize("state_fault", ("drop", "empty"))
def test_full_a_final_gate_withholds_completion_after_durable_ack(
    monkeypatch, tmp_path, state_fault,
):
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=2, num_updates=1,
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
    recipe = module.FULL_MDP_PPO_RECIPE
    assert module._snapshot_indices(
        recipe.max_iterations, recipe.save_interval
    ) == (
        *range(0, recipe.max_iterations, recipe.save_interval),
        recipe.max_iterations - 1,
    )
    monkeypatch.setattr(
        module, "_rsl3_runner",
        lambda: pytest.fail("shape validation must precede RSL construction"),
    )
    expected_shape = (
        f"{recipe.num_envs}x{recipe.num_steps_per_env}x"
        f"{recipe.max_iterations}"
    )
    with pytest.raises(ValueError, match=expected_shape):
        module.main(
            num_envs=2, num_updates=recipe.max_iterations,
            full_a_mode=True, evidence_jsonl=str(tmp_path / "updates.jsonl"),
            snapshot_dir=str(tmp_path), completion_json=str(tmp_path / "seal.json"),
            source_commit=SOURCE_COMMIT, run_namespace=RUN_NAMESPACE,
        )

    cadence_root = tmp_path / "cadence"
    cadence_root.mkdir()
    invoke, _trace, _saved, _evidence, snapshots, _completion = _install_fake_stack(
        monkeypatch, cadence_root, num_envs=1,
        num_updates=2, full_a_mode=True,
    )
    assert invoke() == 0
    assert sorted(path.name for path in snapshots.iterdir()) == [
        "model_0.pt", "model_1.pt",
    ]


def test_full_a_rate_probe_reuses_ledger_without_snapshot_or_completion(
    monkeypatch, capsys, tmp_path,
):
    module = _load()
    invoke, trace, saved, evidence, snapshots, completion = _install_fake_stack(
        monkeypatch, tmp_path, num_envs=1,
        num_updates=module.RATE_PROBE_NUM_UPDATES, full_a_mode=True,
        diagnostic_rate_probe=True,
    )

    assert invoke() == 0
    output = capsys.readouterr().out.splitlines()
    record = json.loads(output[-1].split("=", 1)[1])
    rate = record["rate_probe"]
    assert record["diagnostic_unauthorized"] is True
    assert record["formal_evidence"] is False
    assert record["safety_gate"] is False
    assert record["kind"] == "action_ball_mujoco_full_mdp_h48_rate_probe_v1"
    assert record["schema_version"] == 1
    assert record["source_commit"] == SOURCE_COMMIT
    assert record["namespace"] == RUN_NAMESPACE
    assert record["learning_recipe_sha256"] == (
        module.FULL_MDP_PPO_RECIPE.learning_recipe_sha256()
    )
    assert record["task_lifecycle"] == "full_a_diagnostic_rate_probe"
    assert "engineering_run_complete" not in record
    assert "action_ball_full_mdp_ppo_recipe_sha256" not in record
    assert record["candidate_production_execution_recipe"] == (
        module.FULL_MDP_PPO_RECIPE.execution_recipe()
    )
    assert record["candidate_production_execution_recipe_sha256"] == (
        module.FULL_MDP_PPO_RECIPE_SHA256
    )
    actual_rate_recipe = module._rate_execution_recipe(
        num_envs=1,
        num_steps_per_env=module.NUM_STEPS_PER_ENV,
        max_iterations=module.RATE_PROBE_NUM_UPDATES,
        save_interval=module.FULL_A_SAVE_INTERVAL,
    )
    assert record["rate_execution_recipe"] == actual_rate_recipe
    assert record["rate_execution_recipe_sha256"] == (
        module._canonical_payload_sha256(actual_rate_recipe)
    )
    assert record["rate_execution_recipe_sha256"] != (
        record["candidate_production_execution_recipe_sha256"]
    )
    assert actual_rate_recipe["runner_overrides"] == {
        "num_envs": {"candidate_production": 512, "rate_execution": 1},
        "max_iterations": {
            "candidate_production": 100_000,
            "rate_execution": 61,
        },
    }
    assert rate["warmup_updates"] == 10
    assert rate["measured_updates"] == 50
    assert rate["tail_updates"] == 1
    assert len(rate["measured_update_seconds"]) == 50
    assert rate["measured_transitions"] == 50 * 48
    assert rate["measured_wall_seconds"] > 0
    assert rate["measured_transitions_per_second"] > 0
    assert 0 < rate["update_seconds_p50"] <= rate["update_seconds_p90"]
    assert len(evidence.read_text().splitlines()) == 61
    assert trace.count("prepare") == trace.count("ack") == 61
    assert saved == [] and list(snapshots.iterdir()) == []
    assert not completion.exists()


@pytest.mark.parametrize("profiler_env", (
    "HOPE_ACTION_BALL_UPDATE_PROFILE",
    "HOPE_ACTION_BALL_FULL_MDP_PROFILE_UPDATES",
))
def test_rate_probe_keeps_production_shape_and_rejects_artifact_authority(
    monkeypatch, tmp_path, profiler_env,
):
    module = _load()
    production_rate_recipe = module._rate_execution_recipe()
    assert production_rate_recipe["effective_runner"] == {
        "num_envs": 512,
        "num_steps_per_env": 48,
        "max_iterations": 61,
        "save_interval": 2_000,
    }
    assert production_rate_recipe["runner_overrides"] == {
        "max_iterations": {
            "candidate_production": 100_000,
            "rate_execution": 61,
        }
    }
    assert module._canonical_payload_sha256(production_rate_recipe) != (
        module.FULL_MDP_PPO_RECIPE_SHA256
    )
    monkeypatch.setattr(
        module, "_rsl3_runner",
        lambda: pytest.fail("rate validation must precede RSL construction"),
    )
    common = dict(
        full_a_mode=True, diagnostic_rate_probe=True,
        evidence_jsonl=str(tmp_path / "updates.jsonl"),
        source_commit=SOURCE_COMMIT, run_namespace=RUN_NAMESPACE,
        mujoco_warp_runtime_site=str(tmp_path / "runtime_site"),
    )
    with pytest.raises(ValueError, match="512x48x61"):
        module.main(num_envs=2, num_updates=61, **common)
    with pytest.raises(ValueError, match="forbids snapshot/completion"):
        module.main(
            num_envs=512, num_updates=61,
            snapshot_dir=str(tmp_path / "snapshots"), **common,
        )
    assert profiler_env in module.RATE_PROBE_PROFILE_ENVS
    monkeypatch.setenv(profiler_env, "1")
    with pytest.raises(ValueError, match="profiler environment off"):
        module.main(num_envs=512, num_updates=61, **common)


def test_rate_probe_cli_has_no_resume_surface():
    result = subprocess.run(
        [
            sys.executable,
            str(LANE / "mujoco_gpu_ac_full_mdp_wait_rsl3.py"),
            "--full-a", "--diagnostic-rate-probe", "--resume",
        ],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --resume" in result.stderr


def test_controller_trace_is_bounded_and_cannot_share_training_artifacts(
    monkeypatch, tmp_path,
):
    module = _load()
    monkeypatch.setattr(
        module, "_rsl3_runner",
        lambda: pytest.fail("controller-trace validation must precede RSL construction"),
    )
    common = dict(
        full_a_mode=True,
        num_envs=512,
        diagnostic_controller_trace_checkpoint=str(tmp_path / "model.pt"),
        diagnostic_controller_trace_output=str(tmp_path / "trace"),
        source_commit=SOURCE_COMMIT,
        run_namespace=RUN_NAMESPACE,
        mujoco_warp_runtime_site=str(tmp_path / "runtime_site"),
    )
    with pytest.raises(ValueError, match="exactly 240 steps"):
        module.main(diagnostic_controller_trace_steps=239, **common)
    with pytest.raises(ValueError, match="forbids training evidence"):
        module.main(evidence_jsonl=str(tmp_path / "updates.jsonl"), **common)
    incomplete = dict(common)
    incomplete.pop("diagnostic_controller_trace_checkpoint")
    with pytest.raises(ValueError, match="requires --full-a, checkpoint/output"):
        module.main(**incomplete)


def test_controller_trace_reads_the_live_pd_owner_not_a_copied_law():
    module = _load()
    source = inspect.getsource(module._run_controller_trace)
    main_source = inspect.getsource(module.main)
    plant = (LANE / "a3_train_ppo.py").read_text()
    assert "env.enable_controller_trace()" in source
    assert "trace = env.controller_trace()" in source
    assert 'bits = extras.get("termination_bits")' in source
    assert 'termination_reason_rows["done_without_reason"]' in source
    assert "termination_bit_contract.items()" in source
    assert "termination_bit_contract=wait.FULLMDP_TERMINATION_BITS" in main_source
    assert "if full_a_mode and not controller_trace:" in main_source
    assert "tau_raw = self.kp * (q_des - self._qpos_act()) - self.kd * self._qvel_act()" in plant
    assert 'trace["tau_raw_first"] = tau_raw.clone()' in plant
    assert 'trace["tau_clamped_first"] = tau.clone()' in plant
    assert 'trace["tau_raw_abs_max"].copy_(torch.maximum(' in plant
    assert 'trace["q_min"].copy_(torch.minimum(' in plant


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
        module._write_completion(str(seal), {"schema_version": 5})
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


def _fake_compiled_plant_env():
    expected = _plant_contract().expected_plant_model_identity()["source_plant"]
    return types.SimpleNamespace(
        env=types.SimpleNamespace(xml_path=PLANT_XML),
        _table_keepout=types.SimpleNamespace(
            plant_identity_receipt={
                "root_mjcf_sha256": expected["root_mjcf_sha256"],
                "identity_manifest_sha256": expected["manifest_sha256"],
                "portable_identity_sha256": expected["portable_identity_sha256"],
                "verification_receipt_sha256": VERIFICATION_RECEIPT_SHA256,
                "owner_local_frame_sha256": OWNER_LOCAL_FRAME_SHA256,
            }
        ),
        decimation=20,
        step_dt=0.02,
        sim_cfg=types.SimpleNamespace(
            cone="elliptic",
            add_pairs=True,
            ball_spawn_hope=(2.0, -0.7625, 0.68),
        ),
        njmax_alloc=572,
        naconmax_alloc=128,
        num_envs=1,
        mj_model=types.SimpleNamespace(
            opt=types.SimpleNamespace(
                timestep=0.001,
                integrator=0,
                solver=2,
                cone=1,
                ccd_iterations=35,
                noslip_iterations=0,
            )
        ),
    )


def _bind_runtime_identity_fakes(module, monkeypatch):
    contract = module._plant_contract_module()
    expected = contract.expected_plant_model_identity()["runtime_attach"]
    monkeypatch.setattr(module, "_mujoco_module", lambda: object())
    monkeypatch.setattr(
        module,
        "_geometry_source_identity",
        lambda: expected["geometry_source_sha256"],
    )


def test_compiled_plant_identity_is_exact_and_bound(monkeypatch):
    module = _load()
    _bind_runtime_identity_fakes(module, monkeypatch)
    monkeypatch.setenv("A3_PINGPONG_XML", str(PLANT_XML))
    scan = _source_scan()
    assert module._plant_model_identity(
        _fake_compiled_plant_env(), PLANT_XML, scan, scan, _augmented_mjb()
    ) == _plant_model()


def test_source_closure_scan_matches_the_pinned_manifest_without_compiling():
    module = _load()
    assert module._scan_plant_source(PLANT_XML) == _source_scan()


@pytest.mark.parametrize(
    "field,value",
    (
        ("timestep", 0.002),
        ("decimation", 4),
        ("step_dt", 0.04),
    ),
)
def test_runtime_attachment_rejects_each_policy_clock_drift(
    monkeypatch, field, value,
):
    module = _load()
    _bind_runtime_identity_fakes(module, monkeypatch)
    monkeypatch.setenv("A3_PINGPONG_XML", str(PLANT_XML))
    env = _fake_compiled_plant_env()
    augmented_mjb = _augmented_mjb()
    if field in ("decimation", "step_dt"):
        setattr(env, field, value)
    else:
        # timestep belongs to the serialized MJB. Simulate the changed bytes
        # instead of maintaining a second option-field gate in production.
        augmented_mjb = {
            "relative_locator": "runtime.mjb",
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    with pytest.raises(RuntimeError, match="runtime plant attachment differs"):
        module._plant_model_identity(
            env, PLANT_XML, _source_scan(), _source_scan(), augmented_mjb,
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("njmax", 571),
        ("nconmax", 127),
    ),
)
def test_runtime_attachment_rejects_capacity_drift(
    monkeypatch, field, value,
):
    module = _load()
    _bind_runtime_identity_fakes(module, monkeypatch)
    env = _fake_compiled_plant_env()
    if field == "njmax":
        env.njmax_alloc = value
    else:
        env.naconmax_alloc = value
    with pytest.raises(RuntimeError, match="runtime plant attachment differs"):
        module._plant_model_identity(
            env, PLANT_XML, _source_scan(), _source_scan(), _augmented_mjb()
        )


def test_runtime_attachment_rejects_geometry_mjb_and_owner_drift(monkeypatch):
    module = _load()
    _bind_runtime_identity_fakes(module, monkeypatch)
    env = _fake_compiled_plant_env()
    monkeypatch.setattr(module, "_geometry_source_identity", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="runtime plant attachment differs"):
        module._plant_model_identity(
            env, PLANT_XML, _source_scan(), _source_scan(), _augmented_mjb()
        )

    _bind_runtime_identity_fakes(module, monkeypatch)
    bad_mjb = {
        "relative_locator": "runtime.mjb",
        "sha256": "0" * 64,
        "size_bytes": 1,
    }
    with pytest.raises(RuntimeError, match="runtime plant attachment differs"):
        module._plant_model_identity(
            env, PLANT_XML, _source_scan(), _source_scan(), bad_mjb,
        )

    _bind_runtime_identity_fakes(module, monkeypatch)
    env._table_keepout.plant_identity_receipt["owner_local_frame_sha256"] = "short"
    with pytest.raises(ValueError, match="owner-local frame receipt differs"):
        module._plant_model_identity(
            env, PLANT_XML, _source_scan(), _source_scan(), _augmented_mjb()
        )


def test_full_a_geometry_override_is_rejected_before_court_import(monkeypatch):
    module = _load()
    monkeypatch.setenv("HOPE_GEOMETRY_PY", "/tmp/alternate-geometry.py")
    with pytest.raises(RuntimeError, match="forbids the ambient"):
        module._require_geometry_source_environment()

    monkeypatch.delenv("HOPE_GEOMETRY_PY")
    module._require_geometry_source_environment()


def test_geometry_source_identity_is_the_exact_checkout_module(monkeypatch):
    module = _load()
    contract = module._plant_contract_module()
    court = types.ModuleType("a3_court_env")
    court.__file__ = str(LANE / "a3_court_env.py")
    court.geom = types.SimpleNamespace(
        __source_path__=str(contract.expected_geometry_source_path())
    )
    monkeypatch.setitem(sys.modules, "a3_court_env", court)
    assert module._geometry_source_identity() == (
        contract.TRUSTED_GEOMETRY_SOURCE_SHA256
    )


def test_compiled_plant_identity_rejects_unbound_or_different_env_path(
    monkeypatch, tmp_path,
):
    module = _load()
    _bind_runtime_identity_fakes(module, monkeypatch)
    env = _fake_compiled_plant_env()
    monkeypatch.delenv("A3_PINGPONG_XML", raising=False)
    with pytest.raises(RuntimeError, match="not bound"):
        module._plant_xml_input()
    different = tmp_path / "a3_pingpong.xml"
    different.write_bytes(PLANT_XML.read_bytes())
    monkeypatch.setenv("A3_PINGPONG_XML", str(different))
    with pytest.raises(RuntimeError, match="plant XML identity differs"):
        module._plant_model_identity(
            env, different, _source_scan(), _source_scan(), _augmented_mjb()
        )


def test_compiled_plant_identity_rejects_post_construction_closure_drift(
    monkeypatch,
):
    module = _load()
    _bind_runtime_identity_fakes(module, monkeypatch)
    before, after = _source_scan(), _source_scan()
    after["source_closure_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="plant XML identity differs"):
        module._plant_model_identity(
            _fake_compiled_plant_env(), PLANT_XML, before, after,
            _augmented_mjb(),
        )


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
    assert '"environment_steps": 48' in line
    assert '"transitions": 96' in line
    assert '"policy_width": 229' in line
    assert '"critic_width": 399' in line
