"""Host-only tests for the exact-resume state package (jiayi 9f684ae5 ported to main).

人话:环境的 common_step_counter 驱动所有随步数渐进的课程,但 base rsl_rl 的存档不含它,
续训时全部课程静默回到第 0 步。这里在不起 Isaac 的前提下,用 stub env/runner 验证:
  1. save→load round-trip:课程主时钟 + 各命令项课程状态(EMA 标量、post-swing 环、
     HER 已实现回放环)完整恢复,迭代号跳到 N+1,恢复后 reset 一次;
  2. 老 checkpoint(无状态包)按迭代号精确推算主时钟,而不是回零;
  3. 未知键/未知命令项/未知段一律容忍,认得的照常恢复;
  4. 评测器路径(log_dir=None)保持移植前行为,一个字节都不动环境;
  5. Ctrl-C 等当前 PPO 迭代完整结束才存档退出,存出来的档带状态包。
"""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
import random
import signal
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "source/whole_body_tracking/whole_body_tracking/utils"


def _module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_contract_module():
    spec = importlib.util.spec_from_file_location(
        "training_contract_exact_resume_under_test", UTILS / "training_contract.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeOnPolicyRunner:
    """与真 rsl_rl OnPolicyRunner 的 save/load/learn 契约对齐的最小基类。

    save 真写 torch.save(model/optimizer/iter/infos),load 真读 —— round-trip 必须过真文件,
    否则测不出"状态是否真进了 PT"。learn 模拟 base 的 rollout→update→log 循环,让子类的
    Ctrl-C 边界逻辑有处可挂。
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env = env
        self.cfg = dict(train_cfg or {})
        self.log_dir = log_dir
        self.device = device
        self.logger_type = str(self.cfg.get("logger", "tensorboard"))
        self.writer = None
        self.disable_logs = True
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.num_steps_per_env = int(self.cfg.get("num_steps_per_env", 24))
        self.alg = SimpleNamespace(
            learning_rate=1.0e-3,
            schedule="adaptive",
            optimizer=SimpleNamespace(param_groups=[{"lr": 1.0e-3}]),
            update=lambda: None,
        )
        self.empirical_normalization = False
        self.obs_normalizer = None

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "iter": int(self.current_learning_iteration),
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True, **kwargs):
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        self.current_learning_iteration = int(loaded["iter"])
        return loaded["infos"]

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        start = int(self.current_learning_iteration)
        self.ran_iterations = []
        for it in range(start, start + int(num_learning_iterations)):
            hook = getattr(self, "_test_mid_iteration_hook", None)
            if hook is not None:
                hook(it)  # 模拟 rollout 进行中(Ctrl-C 会在这里到达)
            self.alg.update()
            self.current_learning_iteration = it
            self.ran_iterations.append(it)
            self.log({"it": it})

    def log(self, locs, width=80, pad=35):
        return None


def _load_runner_module(monkeypatch, contract_module):
    fake_rsl_rl = _module("rsl_rl")
    fake_rsl_rl.__path__ = []
    fake_runners = _module("rsl_rl.runners")
    fake_runners.__path__ = []
    fake_isaaclab_rl = _module("isaaclab_rl")
    fake_isaaclab_rl.__path__ = []
    fake_wbt = _module("whole_body_tracking")
    fake_wbt.__path__ = []
    fake_utils = _module("whole_body_tracking.utils")
    fake_utils.__path__ = []
    modules = {
        "rsl_rl": fake_rsl_rl,
        "rsl_rl.env": _module("rsl_rl.env", VecEnv=type("VecEnv", (), {})),
        "rsl_rl.runners": fake_runners,
        "rsl_rl.runners.on_policy_runner": _module(
            "rsl_rl.runners.on_policy_runner", OnPolicyRunner=FakeOnPolicyRunner
        ),
        "isaaclab_rl": fake_isaaclab_rl,
        "isaaclab_rl.rsl_rl": _module(
            "isaaclab_rl.rsl_rl", export_policy_as_onnx=lambda *args, **kwargs: None
        ),
        "whole_body_tracking": fake_wbt,
        "whole_body_tracking.utils": fake_utils,
        "whole_body_tracking.utils.exporter": _module(
            "whole_body_tracking.utils.exporter",
            attach_onnx_metadata=lambda *args, **kwargs: None,
            export_motion_policy_as_onnx=lambda *args, **kwargs: False,
            is_empirical_normalizer=lambda value: False,
        ),
        "whole_body_tracking.utils.training_contract": contract_module,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "motion_runner_exact_resume_under_test", UTILS / "my_on_policy_runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def runner_module(monkeypatch):
    return _load_runner_module(monkeypatch, _load_contract_module())


def test_frozen_eval_hashes_effective_privileged_normalizer(
    runner_module, monkeypatch
):
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.empirical_normalization = True
    runner.obs_normalizer = torch.nn.Linear(3, 3, bias=False)
    runner.privileged_obs_normalizer = torch.nn.Linear(
        5, 5, bias=False
    )
    monkeypatch.setattr(
        runner_module,
        "is_empirical_normalizer",
        lambda value: not isinstance(value, torch.nn.Identity),
    )

    actor = runner._frozen_eval_normalizer_payload("actor")
    critic = runner._frozen_eval_normalizer_payload("critic")
    assert actor["enabled"] is True
    assert critic["enabled"] is True
    critic_sha = runner._frozen_eval_state_binding(critic)["sha256"]

    with torch.no_grad():
        runner.privileged_obs_normalizer.weight[0, 0].add_(1.0)
    changed_sha = runner._frozen_eval_state_binding(
        runner._frozen_eval_normalizer_payload("critic")
    )["sha256"]
    assert changed_sha != critic_sha

    runner.critic_obs_normalizer = torch.nn.Linear(
        5, 5, bias=False
    )
    with pytest.raises(
        RuntimeError, match="normalizer aliases disagree"
    ):
        runner._frozen_eval_normalizer_payload("critic")


def test_frozen_eval_normalizer_binding_fails_closed_when_missing(
    runner_module, monkeypatch
):
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.empirical_normalization = True
    runner.obs_normalizer = torch.nn.Linear(3, 3, bias=False)
    monkeypatch.setattr(
        runner_module,
        "is_empirical_normalizer",
        lambda value: not isinstance(value, torch.nn.Identity),
    )
    with pytest.raises(RuntimeError, match="critic.*absent"):
        runner._frozen_eval_normalizer_payload("critic")

    runner.privileged_obs_normalizer = None
    with pytest.raises(RuntimeError, match="empirical critic.*missing"):
        runner._frozen_eval_normalizer_payload("critic")


# ---------------------------------------------------------------------------
# Stub env:与真 ManagerBasedRLEnv/RslRlVecEnvWrapper 的最小接口对齐
# ---------------------------------------------------------------------------


class FakeCommandManager:
    def __init__(self, terms):
        self._terms = dict(terms)

    @property
    def active_terms(self):
        return list(self._terms)

    def get_term(self, name):
        return self._terms[name]  # 未知项抛 KeyError(与 Isaac 的 dict 语义一致)


class FakeVecEnv:
    def __init__(self, unwrapped):
        self.unwrapped = unwrapped
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        return None, {}


def _motion_term(filled: bool):
    """MotionCommand 的课程宿主子集:自适应失败分箱 + post-swing 收势环形缓冲。"""
    term = SimpleNamespace(device="cpu", metrics={})
    if filled:
        term.bin_failed_count = torch.tensor([3.0, 1.0, 0.5, 2.0])
        term._current_bin_failed = torch.tensor([1.0, 0.0, 0.0, 1.0])
        term._post_swing_root = torch.arange(26, dtype=torch.float32).reshape(2, 13)
        term._post_swing_joint_pos = torch.full((2, 5), 0.25)
        term._post_swing_joint_vel = torch.full((2, 5), -0.5)
        term._post_swing_ptr = 1
        term._post_swing_count = 2
    else:
        term.bin_failed_count = torch.zeros(4)
        term._current_bin_failed = torch.zeros(4)
        term._post_swing_root = None  # 真实类是懒初始化:恢复必须能从 None 起
        term._post_swing_joint_pos = None
        term._post_swing_joint_vel = None
        term._post_swing_ptr = 0
        term._post_swing_count = 0
    # 每 env 瞬态,绝不该入档
    term.time_steps = torch.zeros(4, dtype=torch.long)
    return term


def _racket_term(filled: bool):
    """RacketTargetCommand 的课程宿主子集:EMA 标量、成功门控扩幅、自适应 sigma、HER 回放环。"""
    term = SimpleNamespace(device="cpu", metrics={})
    if filled:
        term._curr_perturb_scale = 0.65
        term._adaptive_sigma_pos = 0.09
        term._adaptive_sigma_vel = 0.8
        term._adaptive_sigma_normal = 0.35
        term._exact_n_acc = 321.5
        term._exact_pass_comp_acc = 123.25
        term._vb_inb_acc_c = {0: 11.0, 1: 7.5}
        term._exact_pos_err_sum = 14.5
        term._exact_pos_err_sum_c = {0: 9.0, 1: 5.5}
        term._exact_nrm_err_sum = 10.5
        term._exact_nrm_err_sum_c = {0: 6.25, 1: 4.25}
        term._ach_fill = {0: 4, 1: 2}
        term._ach_ptr = {0: 0, 1: 2}
        term._ach_pos = {0: torch.ones(4, 3) * 0.1, 1: torch.ones(4, 3) * 0.2}
        term._ach_vel = {0: torch.ones(4, 3) * 1.1, 1: torch.ones(4, 3) * 2.2}
        term._ach_spd = {0: torch.ones(4) * 0.9, 1: torch.ones(4) * 1.3}
    else:
        term._curr_perturb_scale = 0.05
        term._adaptive_sigma_pos = 0.25
        term._adaptive_sigma_vel = 1.8
        term._adaptive_sigma_normal = 0.60
        term._exact_n_acc = 0.0
        term._exact_pass_comp_acc = 0.0
        term._vb_inb_acc_c = {0: 0.0, 1: 0.0}
        term._exact_pos_err_sum = 0.0
        term._exact_pos_err_sum_c = {0: 0.0, 1: 0.0}
        term._exact_nrm_err_sum = 0.0
        term._exact_nrm_err_sum_c = {0: 0.0, 1: 0.0}
        term._ach_fill = {0: 0, 1: 0}
        term._ach_ptr = {0: 0, 1: 0}
        term._ach_pos = {0: torch.zeros(4, 3), 1: torch.zeros(4, 3)}
        term._ach_vel = {0: torch.zeros(4, 3), 1: torch.zeros(4, 3)}
        term._ach_spd = {0: torch.zeros(4), 1: torch.zeros(4)}
    # 入档黑名单探针:cfg 常数、每 env 瞬态张量、名字撞后缀但值是张量的属性
    term._vb_net_x = 1.37
    term._prev_racket_dist = torch.zeros(4)
    term._weird_acc = torch.zeros(4)
    return term


def _make_runner(runner_module, *, log_dir, filled, iteration=0, counter=0):
    terms = {
        "motion": _motion_term(filled),
        "racket_target": _racket_term(filled),
    }
    inner = SimpleNamespace(
        common_step_counter=counter,
        command_manager=FakeCommandManager(terms),
        num_envs=4,
    )
    wrapper = FakeVecEnv(inner)
    runner = runner_module.MotionOnPolicyRunner(
        wrapper,
        {"num_steps_per_env": 24, "max_iterations": 25000, "logger": "tensorboard"},
        log_dir=log_dir,
        device="cpu",
    )
    runner.current_learning_iteration = iteration
    return runner, inner, wrapper, terms


def _make_contract_bound_runner(
    runner_module,
    *,
    log_dir,
    lineage_exact,
):
    terms = {
        "motion": _motion_term(False),
        "racket_target": _racket_term(False),
    }
    inner = SimpleNamespace(
        common_step_counter=0,
        command_manager=FakeCommandManager(terms),
        num_envs=4,
    )
    wrapper = FakeVecEnv(inner)
    runner = runner_module.MotionOnPolicyRunner(
        wrapper,
        {
            "num_steps_per_env": 24,
            "max_iterations": 25000,
            "logger": "tensorboard",
        },
        log_dir=str(log_dir),
        device="cpu",
        training_contract_schema_version=3,
        training_contract_sha256="a" * 64,
        training_contract_lineage_exact=lineage_exact,
        require_exact_resume_state=True,
    )
    runner.alg.policy = torch.nn.Linear(1, 1)
    return runner


def _install_runtime_bootstrap_stubs(
    monkeypatch,
    *,
    runner,
    log_dir,
):
    """Install the narrow post-dump protocol used by runner-only tests."""

    params = Path(log_dir) / "params"
    params.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("training_contract.json", b"contract"),
        ("env.pkl", b"env"),
        ("agent.pkl", b"agent"),
        ("action_ball_frozen_eval_runtime.json", b"identity"),
    ):
        (params / name).write_bytes(payload)
    content = {
        "source": {
            "repo_root": str(Path(log_dir).resolve()),
            "head_commit_oid": "a" * 40,
        },
        "lineage_payload_sha256": "b" * 64,
    }
    document = {
        "schema_version": 1,
        "kind": "action_ball_runtime_bootstrap_receipt_v1",
        "content": content,
        "content_sha256": hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    receipt_path = (
        params / "action_ball_runtime_bootstrap_receipt.json"
    )

    def write_document(value):
        receipt_path.write_text(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="ascii",
        )

    def artifact_receipt(path):
        candidate = Path(path).resolve()
        raw = candidate.read_bytes()
        return {
            "path": str(candidate),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    def strict_read_json(path, *, label):
        del label
        return json.loads(Path(path).read_text(encoding="ascii"))

    def verify_artifact_receipt(value, *, label):
        del label
        if artifact_receipt(value["path"]) != value:
            raise RuntimeError("artifact bytes drifted")

    inbox = _module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_evaluation_inbox",
        artifact_receipt=artifact_receipt,
        strict_read_json=strict_read_json,
        verify_artifact_receipt=verify_artifact_receipt,
    )

    def validate(document_value, **_kwargs):
        return dict(document_value["content"])

    bootstrap = _module(
        "whole_body_tracking.tasks.tracking.mdp.action_ball_runtime_bootstrap",
        TASK_ID="HOPE-PingPong-ActionBall-AgibotA3-v0",
        validate_runtime_bootstrap_receipt_document=validate,
        runtime_bootstrap_lineage_payload_sha256=(
            lambda value: value["lineage_payload_sha256"]
        ),
    )
    tasks = _module("whole_body_tracking.tasks")
    tracking = _module("whole_body_tracking.tasks.tracking")
    mdp = _module("whole_body_tracking.tasks.tracking.mdp")
    for package in (tasks, tracking, mdp):
        package.__path__ = []
    tasks.tracking = tracking
    tracking.mdp = mdp
    mdp.action_ball_evaluation_inbox = inbox
    mdp.action_ball_runtime_bootstrap = bootstrap
    for name, module in (
        ("whole_body_tracking.tasks", tasks),
        ("whole_body_tracking.tasks.tracking", tracking),
        ("whole_body_tracking.tasks.tracking.mdp", mdp),
        (inbox.__name__, inbox),
        (bootstrap.__name__, bootstrap),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    write_document(document)
    runner._strict_exact_resume_target_mode = lambda: "action_ball"
    runner.training_contract_lineage_exact = True
    runner.training_launch_claim_sha256 = "c" * 64
    publication = {
        "content_sha256": document["content_sha256"],
        "artifact_receipt": artifact_receipt(receipt_path),
    }
    return publication, document, write_document


def _strict_resume_checkpoint(*, lineage_exact):
    model = torch.nn.Linear(1, 1)
    return {
        "iter": 4,
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": "a" * 64,
            "training_contract_lineage_exact": lineage_exact,
            "hope_exact_resume_state": {
                "schema_version": 3,
                "next_learning_iteration": 5,
                "tot_timesteps": 384,
                "tot_time": 1.0,
                "algorithm_learning_rate": 1.0e-3,
                "python_random_state": (),
                "numpy_random_state": (),
                "torch_random_state": torch.zeros(1, dtype=torch.uint8),
                "torch_cuda_random_states": [],
                "torch_cuda_device_count": 0,
                "environment_resume_state": {
                    "schema_version": 3,
                    "common_step_counter": 96,
                    "active_term_names": ["motion", "racket_target"],
                    "command_terms": {
                        "motion": {
                            "scalars": {},
                            "tensors": {},
                            "tensor_dicts": {},
                        },
                        "racket_target": {
                            "scalars": {},
                            "tensors": {},
                            "tensor_dicts": {},
                        },
                    },
                },
            },
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
    }


class _ResumeNormalizer(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, width))
        self.register_buffer("_var", torch.ones(1, width))
        self.register_buffer("_std", torch.ones(1, width))
        self.register_buffer("count", torch.tensor(4, dtype=torch.long))


class _ResumeCommandTerm:
    def __init__(self):
        self.validate_calls = 0
        self.load_calls = 0

    def exact_resume_state_dict(self):
        return {"schema_version": 1}

    def validate_exact_resume_state_dict(self, state, *, strict=True):
        assert strict is True
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("command exact state is invalid")
        self.validate_calls += 1

    def load_exact_resume_state_dict(self, state, strict=True):
        self.validate_exact_resume_state_dict(state, strict=strict)
        self.load_calls += 1

    def finalize_action_ball_exact_resume(self):
        return None


class _ResumeActionTerm:
    control_step_action_delay_enabled = False
    action_runtime_state_required = True

    def __init__(self):
        self.validate_calls = 0
        self.load_calls = 0

    def action_delay_exact_resume_state_dict(self):
        return {"schema_version": 1, "token": torch.tensor([1])}

    def validate_action_delay_exact_resume_state_dict(
        self, state, *, strict=True
    ):
        assert strict is True
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "token"}
            or state["schema_version"] != 1
            or not torch.equal(state["token"], torch.tensor([1]))
        ):
            raise ValueError("action exact state is invalid")
        self.validate_calls += 1

    def load_action_delay_exact_resume_state_dict(
        self, state, *, strict=True
    ):
        self.validate_action_delay_exact_resume_state_dict(
            state, strict=strict
        )
        self.load_calls += 1


def _action_ball_211_preflight_fixture(
    runner_module, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        runner_module,
        "is_empirical_normalizer",
        lambda value: isinstance(value, _ResumeNormalizer),
    )
    racket = _ResumeCommandTerm()
    motion = _ResumeCommandTerm()
    action = _ResumeActionTerm()
    commands = FakeCommandManager(
        {"racket_target": racket, "motion": motion}
    )
    actions = FakeCommandManager({"joint_pos": action})
    inner = SimpleNamespace(
        common_step_counter=96,
        command_manager=commands,
        action_manager=actions,
        cfg=SimpleNamespace(
            obs_mode="action_ball_a211",
            commands=SimpleNamespace(
                racket_target=SimpleNamespace(target_mode="action_ball")
            ),
        ),
        num_envs=4,
    )
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.env = FakeVecEnv(inner)
    runner.log_dir = str(tmp_path)
    runner.require_exact_resume_state = True
    runner.training_contract_schema_version = 3
    runner.training_contract_sha256 = "a" * 64
    runner.training_contract_lineage_exact = False
    runner.training_launch_claim_sha256 = None
    runner.empirical_normalization = True
    runner.obs_normalizer = _ResumeNormalizer(211)
    runner.privileged_obs_normalizer = _ResumeNormalizer(319)
    runner.action_ball_a211_trainability_preflight = {
        "actor_width": 211,
        "critic_width": 319,
    }
    runner.action_ball_c211_trainability_preflight = None
    policy = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    policy(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    runner.alg = SimpleNamespace(
        policy=policy,
        optimizer=optimizer,
        rnd=None,
    )
    environment = {
        "schema_version": 4,
        "common_step_counter": 96,
        "active_term_names": ["racket_target", "motion"],
        "command_terms": {
            "racket_target": {
                "capture_mode": "explicit",
                "term_type": (
                    f"{type(racket).__module__}.{type(racket).__qualname__}"
                ),
                "exact_state": {
                    "schema_version": 1,
                    "integrity_sha256": "d" * 64,
                },
            },
            "motion": {
                "capture_mode": "explicit",
                "term_type": (
                    f"{type(motion).__module__}.{type(motion).__qualname__}"
                ),
                "exact_state": {
                    "schema_version": 1,
                    "action_ball_birth": {
                        "shared_racket_state_sha256": "d" * 64,
                    },
                },
            },
        },
        "active_action_term_names": ["joint_pos"],
        "action_terms": {
            "joint_pos": {
                "capture_mode": "explicit_delay",
                "term_type": (
                    f"{type(action).__module__}.{type(action).__qualname__}"
                ),
                "exact_state": {
                    "schema_version": 1,
                    "token": torch.tensor([1]),
                },
            }
        },
    }
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )
    exact = {
        "schema_version": 3,
        "next_learning_iteration": 5,
        "tot_timesteps": 384,
        "tot_time": 1.0,
        "algorithm_learning_rate": 1.0e-3,
        "python_random_state": random.getstate(),
        "numpy_random_state": runner._serialize_numpy_rng_state(
            __import__("numpy").random.get_state()
        ),
        "torch_random_state": torch.get_rng_state(),
        "torch_cuda_random_states": cuda_states,
        "torch_cuda_device_count": len(cuda_states),
        "environment_resume_state": environment,
    }
    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "obs_norm_state_dict": runner.obs_normalizer.state_dict(),
        "privileged_obs_norm_state_dict": (
            runner.privileged_obs_normalizer.state_dict()
        ),
        "iter": 4,
        "infos": {
            "training_contract_schema_version": 3,
            "training_contract_sha256": "a" * 64,
            "training_contract_lineage_exact": 0,
            "hope_exact_resume_state": exact,
        },
    }
    return runner, checkpoint, racket, motion, action


def _mutable_resume_hashes(runner):
    return {
        "policy": runner._exact_resume_tree_sha256(
            runner.alg.policy.state_dict()
        ),
        "optimizer": runner._exact_resume_tree_sha256(
            runner.alg.optimizer.state_dict()
        ),
        "actor": runner._exact_resume_tree_sha256(
            runner.obs_normalizer.state_dict()
        ),
        "critic": runner._exact_resume_tree_sha256(
            runner.privileged_obs_normalizer.state_dict()
        ),
    }


@pytest.mark.parametrize(
    ("live_lineage", "checkpoint_lineage"),
    [(False, 0), (True, 1)],
)
def test_strict_resume_state_is_independent_from_formal_lineage(
    runner_module,
    tmp_path,
    monkeypatch,
    live_lineage,
    checkpoint_lineage,
):
    runner = _make_contract_bound_runner(
        runner_module,
        log_dir=tmp_path,
        lineage_exact=live_lineage,
    )
    assert runner.require_exact_resume_state is True
    assert runner.training_contract_lineage_exact is live_lineage
    monkeypatch.setattr(
        runner,
        "_validated_exact_rng_state",
        lambda state: None,
    )
    monkeypatch.setattr(
        runner,
        "_validate_required_adam_state",
        lambda *args, **kwargs: None,
    )

    checkpoint = _strict_resume_checkpoint(
        lineage_exact=checkpoint_lineage
    )
    state = runner._preflight_required_exact_resume_checkpoint(
        checkpoint,
        path="model_4.pt",
        load_optimizer=True,
    )
    assert state["next_learning_iteration"] == 5

    checkpoint["infos"]["training_contract_lineage_exact"] = (
        1 - checkpoint_lineage
    )
    with pytest.raises(RuntimeError, match="expected lineage"):
        runner._preflight_required_exact_resume_checkpoint(
            checkpoint,
            path="wrong_lineage.pt",
            load_optimizer=True,
        )


def test_action_ball_preflight_accepts_outer3_inner4_and_calls_read_only_hooks(
    runner_module, tmp_path, monkeypatch
):
    runner, checkpoint, racket, motion, action = (
        _action_ball_211_preflight_fixture(
            runner_module, tmp_path, monkeypatch
        )
    )
    before = _mutable_resume_hashes(runner)
    state = runner._preflight_required_exact_resume_checkpoint(
        checkpoint,
        path="model_4.pt",
        load_optimizer=True,
    )
    assert state["schema_version"] == 3
    assert state["environment_resume_state"]["schema_version"] == 4
    assert racket.validate_calls == 1
    assert motion.validate_calls == 1
    assert action.validate_calls == 1
    assert racket.load_calls == motion.load_calls == action.load_calls == 0
    assert _mutable_resume_hashes(runner) == before


def test_action_ball_preflight_rejects_schema3_before_any_live_mutation(
    runner_module, tmp_path, monkeypatch
):
    runner, checkpoint, _racket, _motion, _action = (
        _action_ball_211_preflight_fixture(
            runner_module, tmp_path, monkeypatch
        )
    )
    nested = checkpoint["infos"]["hope_exact_resume_state"][
        "environment_resume_state"
    ]
    nested.pop("active_action_term_names")
    nested.pop("action_terms")
    nested["schema_version"] = 3
    before = _mutable_resume_hashes(runner)
    with pytest.raises(RuntimeError, match="inner action schema 4"):
        runner._preflight_required_exact_resume_checkpoint(
            checkpoint,
            path="schema3.pt",
            load_optimizer=True,
        )
    assert _mutable_resume_hashes(runner) == before


def test_action_ball_preflight_rejects_false_command_resume_declaration_atomically(
    runner_module, tmp_path, monkeypatch
):
    runner, checkpoint, racket, motion, action = (
        _action_ball_211_preflight_fixture(
            runner_module, tmp_path, monkeypatch
        )
    )
    checkpoint["infos"]["hope_exact_resume_state"][
        "environment_resume_state"
    ]["command_terms"]["racket_target"]["exact_state"] = {
        "schema_version": 1,
        "exact_resume_supported": False,
    }
    checkpoint_path = tmp_path / "fresh-only.pt"
    torch.save(checkpoint, checkpoint_path)
    runner._loaded_checkpoint_path = "sentinel-before-rejection"
    before = _mutable_resume_hashes(runner)
    with pytest.raises(
        RuntimeError, match="exact_resume_supported=false"
    ):
        runner.load(str(checkpoint_path), load_optimizer=True)
    assert _mutable_resume_hashes(runner) == before
    assert runner._loaded_checkpoint_path == "sentinel-before-rejection"
    assert racket.validate_calls == motion.validate_calls == 0
    assert action.validate_calls == 0
    assert racket.load_calls == motion.load_calls == action.load_calls == 0


def test_action_ball_preflight_rejects_cross_payload_digest_before_mutation(
    runner_module, tmp_path, monkeypatch
):
    runner, checkpoint, racket, motion, action = (
        _action_ball_211_preflight_fixture(
            runner_module, tmp_path, monkeypatch
        )
    )
    checkpoint["infos"]["hope_exact_resume_state"][
        "environment_resume_state"
    ]["command_terms"]["motion"]["exact_state"]["action_ball_birth"][
        "shared_racket_state_sha256"
    ] = "e" * 64
    before = _mutable_resume_hashes(runner)
    with pytest.raises(RuntimeError, match="cross-payload digest differs"):
        runner._preflight_required_exact_resume_checkpoint(
            checkpoint,
            path="cross_payload_mismatch.pt",
            load_optimizer=True,
        )
    assert _mutable_resume_hashes(runner) == before
    assert racket.validate_calls == motion.validate_calls == 1
    assert action.validate_calls == 0
    assert racket.load_calls == motion.load_calls == action.load_calls == 0


def test_action_ball_preflight_rejects_nonfinite_or_wrong_width_normalizer_atomically(
    runner_module, tmp_path, monkeypatch
):
    runner, checkpoint, racket, motion, action = (
        _action_ball_211_preflight_fixture(
            runner_module, tmp_path, monkeypatch
        )
    )
    checkpoint["obs_norm_state_dict"]["_mean"] = torch.full(
        (1, 211), float("nan")
    )
    before = _mutable_resume_hashes(runner)
    with pytest.raises(RuntimeError, match="actor normalizer.*non-finite"):
        runner._preflight_required_exact_resume_checkpoint(
            checkpoint,
            path="bad_actor_norm.pt",
            load_optimizer=True,
        )
    assert _mutable_resume_hashes(runner) == before
    assert racket.validate_calls == motion.validate_calls == 0
    assert action.validate_calls == 0

    checkpoint["obs_norm_state_dict"] = {
        "_mean": torch.zeros(1, 210),
        "_var": torch.ones(1, 210),
        "_std": torch.ones(1, 210),
        "count": torch.tensor(4, dtype=torch.long),
    }
    with pytest.raises(RuntimeError, match="shape/dtype differs"):
        runner._preflight_required_exact_resume_checkpoint(
            checkpoint,
            path="wrong_actor_width.pt",
            load_optimizer=True,
        )
    assert _mutable_resume_hashes(runner) == before


def test_formal_apply_uses_the_same_supported_normalizer_aliases_as_preflight(
    runner_module, tmp_path, monkeypatch
):
    runner, checkpoint, _racket, _motion, _action = (
        _action_ball_211_preflight_fixture(
            runner_module, tmp_path, monkeypatch
        )
    )
    actor = runner.obs_normalizer
    critic = runner.privileged_obs_normalizer
    del runner.obs_normalizer
    del runner.privileged_obs_normalizer
    runner.actor_obs_normalizer = actor
    runner.critic_obs_normalizer = critic

    runner._preflight_required_exact_resume_checkpoint(
        checkpoint,
        path="alternate_aliases.pt",
        load_optimizer=True,
    )
    infos = runner._apply_formal_preloaded_checkpoint(
        checkpoint,
        load_optimizer=True,
        prefix="alternate aliases",
    )
    assert infos is checkpoint["infos"]
    assert runner.current_learning_iteration == 4
    assert runner.actor_obs_normalizer is actor
    assert runner.critic_obs_normalizer is critic


def test_diagnostic_strict_runner_saves_lineage_zero(
    runner_module,
    tmp_path,
):
    runner = _make_contract_bound_runner(
        runner_module,
        log_dir=tmp_path,
        lineage_exact=False,
    )
    checkpoint = tmp_path / "diagnostic.pt"
    runner.save(str(checkpoint))
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert loaded["infos"]["training_contract_schema_version"] == 3
    assert loaded["infos"]["training_contract_sha256"] == "a" * 64
    assert loaded["infos"]["training_contract_lineage_exact"] == 0
    assert "hope_exact_resume_state" in loaded["infos"]


def test_diagnostic_action_ball_does_not_claim_formal_bootstrap(
    runner_module,
    tmp_path,
):
    runner = _make_contract_bound_runner(
        runner_module,
        log_dir=tmp_path,
        lineage_exact=False,
    )
    runner._strict_exact_resume_target_mode = lambda: "action_ball"
    runner._validate_task_first_exact_resume_terms = lambda: None
    runner._capture_environment_resume_state = lambda: {
        "schema_version": 3,
        "common_step_counter": 0,
        "active_term_names": [],
        "command_terms": {},
    }
    checkpoint = tmp_path / "diagnostic_action_ball.pt"
    runner.save(str(checkpoint))
    infos = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )["infos"]
    assert infos["training_contract_lineage_exact"] == 0
    assert "runtime_bootstrap_receipt_sha256" not in infos
    assert (
        "runtime_bootstrap_receipt_sha256"
        not in infos["hope_exact_resume_state"]
    )


def test_runner_rejects_truthy_non_boolean_lineage(
    runner_module,
    tmp_path,
):
    with pytest.raises(TypeError, match="exact bool"):
        _make_contract_bound_runner(
            runner_module,
            log_dir=tmp_path,
            lineage_exact="0",
        )


def test_action_ball_bootstrap_binding_is_in_checkpoint_and_exact_state(
    runner_module,
    tmp_path,
    monkeypatch,
):
    runner, _, _, _ = _make_runner(
        runner_module,
        log_dir=str(tmp_path),
        filled=False,
    )
    publication, _document, _write = (
        _install_runtime_bootstrap_stubs(
            monkeypatch,
            runner=runner,
            log_dir=tmp_path,
        )
    )
    runner.bind_runtime_bootstrap_receipt(**publication)
    runner._validate_task_first_exact_resume_terms = lambda: None
    runner._capture_environment_resume_state = lambda: {
        "schema_version": 3,
        "common_step_counter": 0,
        "active_term_names": [],
        "command_terms": {},
    }

    infos = runner._checkpoint_infos()
    for key in (
        "runtime_bootstrap_receipt_sha256",
        "runtime_bootstrap_lineage_payload_sha256",
        "runtime_bootstrap_receipt",
    ):
        assert infos[key] == infos["hope_exact_resume_state"][key]
    assert (
        infos["runtime_bootstrap_receipt_sha256"]
        == publication["content_sha256"]
    )


def test_action_ball_bootstrap_binding_detects_receipt_replacement(
    runner_module,
    tmp_path,
    monkeypatch,
):
    runner, _, _, _ = _make_runner(
        runner_module,
        log_dir=str(tmp_path),
        filled=False,
    )
    publication, document, write_document = (
        _install_runtime_bootstrap_stubs(
            monkeypatch,
            runner=runner,
            log_dir=tmp_path,
        )
    )
    runner.bind_runtime_bootstrap_receipt(**publication)
    replaced = dict(document)
    replaced["content"] = dict(document["content"])
    replaced["content"]["lineage_payload_sha256"] = "d" * 64
    replaced["content_sha256"] = "e" * 64
    write_document(replaced)
    with pytest.raises(RuntimeError, match="bytes drifted"):
        runner._validated_runtime_bootstrap_binding()


def test_action_ball_load_rejects_infos_exact_bootstrap_mismatch(
    runner_module,
    tmp_path,
    monkeypatch,
):
    runner, _, _, _ = _make_runner(
        runner_module,
        log_dir=str(tmp_path),
        filled=False,
    )
    publication, _document, _write = (
        _install_runtime_bootstrap_stubs(
            monkeypatch,
            runner=runner,
            log_dir=tmp_path,
        )
    )
    runner.bind_runtime_bootstrap_receipt(**publication)
    binding = runner._validated_runtime_bootstrap_binding()
    mismatched = dict(binding)
    mismatched["runtime_bootstrap_lineage_payload_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="infos/exact-state"):
        runner._validate_checkpoint_runtime_bootstrap_binding(
            checkpoint_infos=binding,
            exact_state=mismatched,
            prefix="test checkpoint",
        )


def test_exact_resume_roundtrip_save_preserves_loaded_iteration_and_receipt(
    runner_module,
    tmp_path,
    monkeypatch,
):
    runner, _, _, _ = _make_runner(
        runner_module,
        log_dir=str(tmp_path),
        filled=False,
        iteration=8,
    )
    publication, _document, _write = (
        _install_runtime_bootstrap_stubs(
            monkeypatch,
            runner=runner,
            log_dir=tmp_path,
        )
    )
    runner.bind_runtime_bootstrap_receipt(**publication)
    runner._validate_task_first_exact_resume_terms = lambda: None
    live_common_step = [192]
    runner._capture_environment_resume_state = lambda: {
        "schema_version": 3,
        "common_step_counter": live_common_step[0],
        "active_term_names": [],
        "command_terms": {},
    }
    runner.current_learning_iteration = 7
    source_state = runner._build_exact_resume_state()
    source_state["wandb_run_id"] = "source-wandb-id"
    source_state["wandb_run_name"] = "source-wandb-name"
    runner.current_learning_iteration = 8
    runner._exact_resume_loaded_source_iteration = 7
    runner._exact_resume_loaded_source_telemetry = {
        key: source_state[key]
        for key in ("log_dir", "wandb_run_id", "wandb_run_name")
    }
    runner.alg.policy = torch.nn.Linear(1, 1)
    runner.alg.optimizer = torch.optim.Adam(
        runner.alg.policy.parameters(), lr=1.0e-3
    )
    runner.privileged_obs_normalizer = None
    runner._exact_resume_roundtrip_pending = True
    runner._action_ball_resume_reset_pending = True
    runner._exact_resume_loaded_source_common_step_counter = 192
    runner._exact_resume_live_state_baseline = (
        runner._capture_exact_resume_live_state_content()
    )
    stable_receipt = runner.exact_resume_live_state_receipt()
    assert stable_receipt["content"]["common_step_counter_delta"] == 0

    original_weight = runner.alg.policy.weight.detach().clone()
    with torch.no_grad():
        runner.alg.policy.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="model_state_sha256"):
        runner.exact_resume_live_state_receipt()
    with torch.no_grad():
        runner.alg.policy.weight.copy_(original_weight)

    torch_rng = torch.get_rng_state()
    torch.rand(1)
    with pytest.raises(RuntimeError, match="rng_state_sha256"):
        runner.exact_resume_live_state_receipt()
    torch.set_rng_state(torch_rng)

    live_common_step[0] += 1
    with pytest.raises(RuntimeError, match="step/reset"):
        runner.exact_resume_live_state_receipt()
    live_common_step[0] -= 1
    assert runner.exact_resume_live_state_receipt() == stable_receipt
    target = tmp_path / "roundtrip.pt"

    receipt = runner.save_exact_resume_roundtrip(target)
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["iter"] == 7
    assert (
        loaded["infos"]["hope_exact_resume_state"][
            "next_learning_iteration"
        ]
        == 8
    )
    assert (
        loaded["infos"]["hope_exact_resume_state"]["wandb_run_id"]
        == "source-wandb-id"
    )
    assert (
        loaded["infos"]["hope_exact_resume_state"]["wandb_run_name"]
        == "source-wandb-name"
    )
    assert runner.current_learning_iteration == 8
    assert receipt["source_embedded_iteration"] == 7
    assert receipt["before_current_learning_iteration"] == 8
    assert receipt["after_current_learning_iteration"] == 8
    assert (
        receipt["runtime_bootstrap_receipt"]
        == loaded["infos"]["runtime_bootstrap_receipt"]
    )
    with pytest.raises(RuntimeError, match="fresh strict load"):
        runner.save_exact_resume_roundtrip(tmp_path / "second.pt")


def test_numpy_rng_safe_schema_survives_weights_only_checkpoint_and_restores(
    runner_module,
    tmp_path,
):
    runner, _, _, _ = _make_runner(
        runner_module,
        log_dir=str(tmp_path),
        filled=False,
    )
    python_before = random.getstate()
    numpy_before = __import__("numpy").random.get_state()
    torch_before = torch.get_rng_state()
    try:
        random.seed(713)
        __import__("numpy").random.seed(713)
        torch.manual_seed(713)
        exact_state = runner._build_exact_resume_state()
        assert set(exact_state["numpy_random_state"]) == {
            "schema_version",
            "bit_generator",
            "state_uint32",
            "position",
            "has_gauss",
            "cached_gaussian",
        }
        expected_numpy = __import__("numpy").random.random_sample(8)
        checkpoint = {
            "model_state_dict": {"weight": torch.ones(2, 2)},
            "optimizer_state_dict": {
                "state": {
                    0: {
                        "step": torch.tensor(1.0),
                        "exp_avg": torch.zeros(2, 2),
                        "exp_avg_sq": torch.ones(2, 2),
                    }
                },
                "param_groups": [{"params": [0], "lr": 1.0e-3}],
            },
            "obs_norm_state_dict": {
                "mean": torch.zeros(2),
                "var": torch.ones(2),
            },
            "privileged_obs_norm_state_dict": {
                "mean": torch.zeros(3),
                "var": torch.ones(3),
            },
            "iter": 0,
            "infos": {"hope_exact_resume_state": exact_state},
        }
        stream = io.BytesIO()
        torch.save(checkpoint, stream)
        stream.seek(0)
        decoded = torch.load(
            stream, map_location="cpu", weights_only=True
        )
        __import__("numpy").random.seed(1)
        runner._restore_exact_rng_state(
            decoded["infos"]["hope_exact_resume_state"]
        )
        actual_numpy = __import__("numpy").random.random_sample(8)
        assert __import__("numpy").array_equal(
            actual_numpy, expected_numpy
        )
    finally:
        random.setstate(python_before)
        __import__("numpy").random.set_state(numpy_before)
        torch.set_rng_state(torch_before)


def test_formal_immutable_loader_rejects_reduce_without_execution(
    runner_module,
    tmp_path,
):
    marker = tmp_path / "reduce-executed"

    class Malicious:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    stream = io.BytesIO()
    torch.save({"payload": Malicious()}, stream)
    raw = stream.getvalue()
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.require_exact_resume_state = True
    runner._formal_action_ball_runtime_bootstrap_required = lambda: True
    with pytest.raises(Exception):
        runner.load_formal_action_ball_checkpoint_bytes(
            raw,
            checkpoint_path=tmp_path / "admitted.pt",
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            expected_size_bytes=len(raw),
        )
    assert not marker.exists()


def test_formal_immutable_loader_never_reopens_checkpoint_path(
    runner_module,
    tmp_path,
):
    marker = tmp_path / "path-reduce-executed"

    class Malicious:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))

    path = tmp_path / "swapped.pt"
    torch.save({"payload": Malicious()}, path)
    safe_stream = io.BytesIO()
    torch.save({}, safe_stream)
    admitted = safe_stream.getvalue()
    runner = runner_module.MotionOnPolicyRunner.__new__(
        runner_module.MotionOnPolicyRunner
    )
    runner.require_exact_resume_state = True
    runner._formal_action_ball_runtime_bootstrap_required = lambda: True
    runner._validate_task_first_exact_resume_terms = lambda: None

    def stop_after_safe_decode(*_args, **_kwargs):
        raise RuntimeError("safe preflight reached")

    runner._preflight_required_exact_resume_checkpoint = (
        stop_after_safe_decode
    )
    with pytest.raises(RuntimeError, match="safe preflight reached"):
        runner.load_formal_action_ball_checkpoint_bytes(
            admitted,
            checkpoint_path=path,
            expected_sha256=hashlib.sha256(admitted).hexdigest(),
            expected_size_bytes=len(admitted),
        )
    assert not marker.exists()


# ---------------------------------------------------------------------------
# 1) round-trip:save 打包 → load 恢复
# ---------------------------------------------------------------------------


def test_save_embeds_state_and_load_round_trips(runner_module, tmp_path):
    save_dir = tmp_path / "run_a"
    save_dir.mkdir()
    runner, inner, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=137, counter=3288
    )
    runner.tot_timesteps = 137 * 24 * 4
    runner.tot_time = 456.75
    runner.alg.learning_rate = 3.3e-4
    checkpoint = save_dir / "model_137.pt"
    runner.save(str(checkpoint))

    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded["infos"]["hope_exact_resume_state"]
    assert state["schema_version"] == 3
    assert state["next_learning_iteration"] == 138
    assert state["target_learning_iterations"] == 25000
    assert state["tot_timesteps"] == 137 * 24 * 4
    assert state["algorithm_learning_rate"] == pytest.approx(3.3e-4)
    env_state = state["environment_resume_state"]
    assert env_state["common_step_counter"] == 3288
    racket_saved = env_state["command_terms"]["racket_target"]
    assert racket_saved["scalars"]["_curr_perturb_scale"] == pytest.approx(0.65)
    assert racket_saved["scalars"]["_vb_inb_acc_c"] == {0: 11.0, 1: 7.5}
    assert racket_saved["scalars"]["_exact_pos_err_sum"] == pytest.approx(14.5)
    assert racket_saved["scalars"]["_adaptive_sigma_normal"] == pytest.approx(
        0.35
    )
    assert racket_saved["scalars"]["_exact_nrm_err_sum"] == pytest.approx(10.5)
    assert racket_saved["scalars"]["_exact_nrm_err_sum_c"] == {
        0: 6.25,
        1: 4.25,
    }
    assert racket_saved["scalars"]["_ach_fill"] == {0: 4, 1: 2}
    assert torch.equal(racket_saved["tensor_dicts"]["_ach_pos"][1], torch.ones(4, 3) * 0.2)
    motion_saved = env_state["command_terms"]["motion"]
    assert torch.equal(motion_saved["tensors"]["bin_failed_count"], torch.tensor([3.0, 1.0, 0.5, 2.0]))
    assert motion_saved["scalars"]["_post_swing_ptr"] == 1
    # 入档黑名单:cfg 常数/每 env 瞬态/张量伪标量都不许进
    assert "_vb_net_x" not in racket_saved["scalars"]
    assert "_weird_acc" not in racket_saved["scalars"]
    assert "_prev_racket_dist" not in racket_saved["tensors"]
    assert "time_steps" not in motion_saved["tensors"]

    # 全新进程语义:另起 runner/env(全零课程),从档里恢复
    resume_dir = tmp_path / "run_b"
    resume_dir.mkdir()
    fresh, fresh_inner, fresh_wrapper, fresh_terms = _make_runner(
        runner_module, log_dir=str(resume_dir), filled=False
    )
    fresh.load(str(checkpoint))

    assert fresh.current_learning_iteration == 138  # N+1:不重复第 137 个更新
    assert fresh_inner.common_step_counter == 3288
    assert fresh_wrapper.reset_calls == 1  # 恢复后 reset 一次,rollout 按恢复分布采样
    assert fresh.tot_timesteps == 137 * 24 * 4
    assert fresh.tot_time == pytest.approx(456.75)
    assert fresh.alg.learning_rate == pytest.approx(3.3e-4)  # adaptive 调度:lr 续上
    assert fresh.alg.optimizer.param_groups[0]["lr"] == pytest.approx(3.3e-4)
    racket = fresh_terms["racket_target"]
    assert racket._curr_perturb_scale == pytest.approx(0.65)
    assert racket._adaptive_sigma_pos == pytest.approx(0.09)
    assert racket._adaptive_sigma_normal == pytest.approx(0.35)
    assert racket._exact_n_acc == pytest.approx(321.5)
    assert racket._vb_inb_acc_c == {0: 11.0, 1: 7.5}
    assert racket._exact_pos_err_sum_c == {0: 9.0, 1: 5.5}
    assert racket._exact_nrm_err_sum == pytest.approx(10.5)
    assert racket._exact_nrm_err_sum_c == {0: 6.25, 1: 4.25}
    assert racket._ach_fill == {0: 4, 1: 2}
    assert torch.equal(racket._ach_pos[0], torch.ones(4, 3) * 0.1)
    assert torch.equal(racket._ach_spd[1], torch.ones(4) * 1.3)
    motion = fresh_terms["motion"]
    assert torch.equal(motion.bin_failed_count, torch.tensor([3.0, 1.0, 0.5, 2.0]))
    assert torch.equal(motion._post_swing_root, torch.arange(26, dtype=torch.float32).reshape(2, 13))
    assert motion._post_swing_count == 2
    # 瞬态未被档里的值污染
    assert racket._vb_net_x == pytest.approx(1.37)


def test_fixed_lr_schedule_keeps_config_lr(runner_module, tmp_path):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=10, counter=240
    )
    runner.alg.learning_rate = 9.9e-4
    checkpoint = save_dir / "model_10.pt"
    runner.save(str(checkpoint))

    fresh, _, _, _ = _make_runner(runner_module, log_dir=str(save_dir), filled=False)
    fresh.alg.schedule = "fixed"
    fresh.load(str(checkpoint))
    # 固定调度:YAML 里改 lr 再续训必须生效,不吃档里的旧 lr
    assert fresh.alg.learning_rate == pytest.approx(1.0e-3)
    assert fresh.alg.optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)


# ---------------------------------------------------------------------------
# 2) 老 checkpoint:无状态包,按迭代号精确推算主时钟
# ---------------------------------------------------------------------------


def test_legacy_checkpoint_derives_counter_from_iteration(runner_module, tmp_path):
    checkpoint = tmp_path / "model_507.pt"
    torch.save(
        {"model_state_dict": {}, "optimizer_state_dict": {}, "iter": 507, "infos": None},
        checkpoint,
    )
    fresh, inner, wrapper, terms = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=False
    )
    fresh.load(str(checkpoint))
    # base 存档发生在完成第 507 个迭代之后:主时钟 = (507+1) * num_steps_per_env
    assert inner.common_step_counter == 508 * 24
    # 迭代号维持 base 语义(没有状态包时不敢替它做 N+1 的决定)
    assert fresh.current_learning_iteration == 507
    assert wrapper.reset_calls == 1
    # 课程细节无从恢复:保持新鲜初始化
    assert terms["racket_target"]._curr_perturb_scale == pytest.approx(0.05)


def test_hitterobs_top_level_state_key_is_also_accepted(runner_module, tmp_path):
    """跨栈对账:jiayi/hitterobs 把状态放 PT 顶层而不是 infos 里,也要认。"""
    checkpoint = tmp_path / "model_88.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 88,
            "infos": None,
            "hope_exact_resume_state": {
                "schema_version": 1,  # hitterobs v1:还没有 environment_resume_state
                "next_learning_iteration": 89,
                "tot_timesteps": 88 * 24 * 4,
                "tot_time": 12.0,
            },
        },
        checkpoint,
    )
    fresh, inner, _, _ = _make_runner(runner_module, log_dir=str(tmp_path), filled=False)
    fresh.load(str(checkpoint))
    assert fresh.current_learning_iteration == 89
    # v1 没有环境状态段 → 主时钟按 next_learning_iteration 推算
    assert inner.common_step_counter == 89 * 24


# ---------------------------------------------------------------------------
# 3) 未知键/未知项容忍;未知 schema fail-loud
# ---------------------------------------------------------------------------


def test_unknown_keys_terms_and_sections_are_tolerated(runner_module, tmp_path):
    checkpoint = tmp_path / "model_11.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 11,
            "infos": {
                "hope_exact_resume_state": {
                    "schema_version": 2,
                    "next_learning_iteration": 12,
                    "future_top_level_field": {"nested": True},
                    "environment_resume_state": {
                        "schema_version": 2,
                        "common_step_counter": 777,
                        "future_section": [1, 2, 3],
                        "command_terms": {
                            "ghost_term": {"scalars": {"_ghost_acc": 1.0}},
                            "racket_target": {
                                "scalars": {
                                    "_curr_perturb_scale": 0.44,
                                    "_attr_this_build_lacks_acc": 9.9,
                                },
                                "tensors": {"_tensor_this_build_lacks": torch.zeros(3)},
                                "tensor_dicts": {
                                    "_ach_pos": {
                                        0: torch.full((4, 3), 0.7),
                                        99: torch.full((4, 3), -1.0),  # 档里多出的 clip 键
                                    },
                                    "_dict_this_build_lacks": {0: torch.zeros(2)},
                                },
                                "future_subsection": {"z": 1},
                            },
                        },
                    },
                }
            },
        },
        checkpoint,
    )
    fresh, inner, _, terms = _make_runner(runner_module, log_dir=str(tmp_path), filled=False)
    fresh.load(str(checkpoint))
    assert fresh.current_learning_iteration == 12
    assert inner.common_step_counter == 777
    racket = terms["racket_target"]
    assert racket._curr_perturb_scale == pytest.approx(0.44)
    assert not hasattr(racket, "_attr_this_build_lacks_acc")
    assert not hasattr(racket, "_tensor_this_build_lacks")
    assert torch.equal(racket._ach_pos[0], torch.full((4, 3), 0.7))
    assert 99 not in racket._ach_pos  # 多出的 clip 键被容忍丢弃,不塞进当前配置
    assert not hasattr(racket, "_dict_this_build_lacks")


def test_iter_surgery_invalidates_stale_state_and_falls_back(runner_module, tmp_path, capsys):
    """make_hitter_warmstart / warm_start_realsensor 把 iter 归零但保留 infos:状态包与 iter
    失配时必须响亮降级成'老档'语义,不许劫持刻意的全新热启动。"""
    save_dir = tmp_path / "donor"
    save_dir.mkdir()
    donor, _, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=5000, counter=5001 * 24
    )
    checkpoint = save_dir / "model_5000.pt"
    donor.save(str(checkpoint))
    # 模拟外科手术:iter 归零、丢优化器,infos(含状态包)整份保留
    surgical = torch.load(checkpoint, map_location="cpu", weights_only=False)
    surgical.pop("optimizer_state_dict", None)
    surgical["iter"] = 0
    warmstart = tmp_path / "warmstart.pt"
    torch.save(surgical, warmstart)

    fresh, inner, wrapper, terms = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=False
    )
    fresh.load(str(warmstart))
    assert "hope_exact_resume_state is stale" in capsys.readouterr().out
    assert fresh.current_learning_iteration == 0  # 手术意图(全新热启动)得到尊重
    assert inner.common_step_counter == 1 * 24  # 主时钟按本档 iter 推算,而不是上一世的 5001*24
    assert terms["racket_target"]._curr_perturb_scale == pytest.approx(0.05)  # 课程从头攒
    assert wrapper.reset_calls == 1


def test_unknown_schema_version_fails_loud(runner_module, tmp_path):
    checkpoint = tmp_path / "model_5.pt"
    torch.save(
        {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "iter": 5,
            "infos": {"hope_exact_resume_state": {"schema_version": 99}},
        },
        checkpoint,
    )
    fresh, _, _, _ = _make_runner(runner_module, log_dir=str(tmp_path), filled=False)
    with pytest.raises(RuntimeError, match="hope_exact_resume_state schema"):
        fresh.load(str(checkpoint))


# ---------------------------------------------------------------------------
# 4) 评测器路径(log_dir=None)行为逐字节不变
# ---------------------------------------------------------------------------


def test_eval_runner_without_log_dir_keeps_legacy_load_semantics(runner_module, tmp_path):
    save_dir = tmp_path / "run"
    save_dir.mkdir()
    trainer, _, _, _ = _make_runner(
        runner_module, log_dir=str(save_dir), filled=True, iteration=42, counter=1032
    )
    checkpoint = save_dir / "model_42.pt"
    trainer.save(str(checkpoint))

    evaluator, inner, wrapper, terms = _make_runner(
        runner_module, log_dir=None, filled=False
    )
    evaluator.load(str(checkpoint))
    # isaac_bank_exam / play 语义:不动环境、不 reset、迭代号维持 base load 的值
    assert inner.common_step_counter == 0
    assert wrapper.reset_calls == 0
    assert evaluator.current_learning_iteration == 42
    assert terms["racket_target"]._curr_perturb_scale == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 5) Ctrl-C:等当前迭代完整结束才存档退出
# ---------------------------------------------------------------------------


def test_sigint_defers_to_iteration_boundary_and_saves_resumable_pt(runner_module, tmp_path):
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=True, iteration=0, counter=24
    )
    original_handler = signal.getsignal(signal.SIGINT)

    def _interrupt_mid_iteration(it):
        if it == 0:
            os.kill(os.getpid(), signal.SIGINT)  # rollout 中途按下 Ctrl-C

    runner._test_mid_iteration_hook = _interrupt_mid_iteration
    with pytest.raises(KeyboardInterrupt):
        runner.learn(num_learning_iterations=3)

    # 第 0 个迭代完整跑完并存档;第 1、2 个不再运行
    assert runner.ran_iterations == [0]
    boundary_checkpoint = tmp_path / "model_0.pt"
    assert boundary_checkpoint.is_file()
    loaded = torch.load(boundary_checkpoint, map_location="cpu", weights_only=False)
    state = loaded["infos"]["hope_exact_resume_state"]
    assert state["next_learning_iteration"] == 1
    assert state["environment_resume_state"]["common_step_counter"] == 24
    # 信号处理器恢复原状,进程退出行为不被劫持
    assert signal.getsignal(signal.SIGINT) is original_handler


def test_frozen_eval_two_update_checkpoints_name_completed_policy(
    runner_module, tmp_path
):
    """The update wrapper must not serialize RSL-RL's stale iteration field."""

    runner, _, _, _ = _make_runner(
        runner_module,
        log_dir=str(tmp_path),
        filled=True,
        iteration=0,
        counter=24,
    )
    optimizer_updates = []
    runner.alg.update = lambda: optimizer_updates.append(
        len(optimizer_updates)
    )
    runner._effective_reward_activation_task_kind = lambda: None
    runner._rollout_update_wrapper_active = False

    class AutoAckTerm:
        def __init__(self):
            self.last_step = -1
            self.last_seq = -1
            self.snapshots = []

        def action_ball_frozen_evaluation_boundary(
            self,
            *,
            step,
            phase,
            checkpoint_path="",
            runner_bindings=None,
            **_kwargs,
        ):
            if phase == "poll":
                due = step > self.last_step
                return {
                    "request_seq": (
                        self.last_seq + 1 if due else self.last_seq
                    ),
                    "request_due": due,
                    "needs_global_reset": False,
                    "needs_ack_checkpoint": False,
                    "stage": None if due else "acked",
                    "requires_runner_binding": False,
                }
            if phase == "publish_request":
                loaded = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
                self.snapshots.append(
                    {
                        "step": step,
                        "runner_generation": runner_bindings[
                            "policy_generation"
                        ],
                        "checkpoint_iter": loaded["iter"],
                        "next_learning_iteration": loaded["infos"][
                            "hope_exact_resume_state"
                        ]["next_learning_iteration"],
                    }
                )
                self.last_step = step
                self.last_seq += 1
                return {
                    "request_seq": self.last_seq,
                    "published": True,
                }
            raise AssertionError(
                f"unexpected frozen-eval phase {phase!r}"
            )

    term = AutoAckTerm()
    runner._action_ball_frozen_eval_term = lambda: term
    runner._frozen_eval_runner_bindings = (
        lambda *, policy_generation: {
            "policy_generation": policy_generation,
        }
    )

    runner.learn(num_learning_iterations=2)

    assert optimizer_updates == [0, 1]
    assert term.snapshots == [
        {
            "step": 0,
            "runner_generation": 0,
            "checkpoint_iter": 0,
            "next_learning_iteration": 1,
        },
        {
            "step": 1,
            "runner_generation": 1,
            "checkpoint_iter": 1,
            "next_learning_iteration": 2,
        },
    ]
    assert runner.current_learning_iteration == 1


def test_sigterm_is_also_deferred(runner_module, tmp_path):
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=True, iteration=7, counter=8 * 24
    )
    original_handler = signal.getsignal(signal.SIGTERM)

    def _terminate_mid_iteration(it):
        if it == 7:
            os.kill(os.getpid(), signal.SIGTERM)

    runner._test_mid_iteration_hook = _terminate_mid_iteration
    with pytest.raises(KeyboardInterrupt):
        runner.learn(num_learning_iterations=5)
    assert runner.ran_iterations == [7]
    assert (tmp_path / "model_7.pt").is_file()
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_second_sigint_escalates_to_immediate_default_action(
    runner_module, tmp_path, monkeypatch, capsys
):
    """第二次 Ctrl-C = 立即死(2026-07-25):恢复 SIG_DFL 并对自己重投该信号,不再静默吞掉。

    真升级会当场杀掉测试进程,所以把 os.kill 换成记录桩(monkeypatch 会还原);投递前两次
    信号用真 kill 的保存引用。断言:升级时点上 SIGINT 的处置已经是 SIG_DFL(即使桩没真杀,
    内核默认动作已就位),且提示了"不存档"。
    """
    runner, _, _, _ = _make_runner(
        runner_module, log_dir=str(tmp_path), filled=True, iteration=0, counter=24
    )
    real_kill = os.kill
    escalations = []

    def _recording_kill(pid, signum):
        escalations.append((pid, signum, signal.getsignal(signal.SIGINT)))

    def _double_interrupt(it):
        if it == 0:
            real_kill(os.getpid(), signal.SIGINT)  # 第一次:登记"想停"
            monkeypatch.setattr(os, "kill", _recording_kill)
            real_kill(os.getpid(), signal.SIGINT)  # 第二次:应升级为立即终止

    runner._test_mid_iteration_hook = _double_interrupt
    with pytest.raises(KeyboardInterrupt):  # 桩没真杀,流程走到边界照旧收尾
        runner.learn(num_learning_iterations=3)

    assert escalations == [(os.getpid(), signal.SIGINT, signal.SIG_DFL)]
    out = capsys.readouterr().out
    assert "second interrupt" in out and "NO checkpoint" in out
