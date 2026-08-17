"""Focused CPU tests for the compact ActionEpoch observation manager ABI."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "tasks" / "tracking" / "mdp"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("action_ball_full_mdp_epoch", MDP / "action_ball_full_mdp_epoch.py")
R = _load("action_ball_full_mdp_lean_rewards", MDP / "action_ball_full_mdp_lean_rewards.py")
L = _load("action_ball_full_mdp_lean_runtime", MDP / "action_ball_full_mdp_lean_runtime.py")
O = _load("action_ball_full_mdp_lean_observation_cfg", MDP / "action_ball_full_mdp_lean_observation_cfg.py")


class _Env:
    def __init__(self):
        self._action_ball_full_mdp_manager_construction_state = "runtime_graph_ready"
        self.common_step_counter = 10
        self.step_dt = 0.02

    def _action_ball_full_mdp_lean_observe_term(self, *, group):
        source = getattr(self, "_installed_lean_observation_source", None)
        if (
            type(source) is not O.LeanActionEpochObservationSource
            or source._env is not self
        ):
            raise O.LeanObservationConstructionHold(
                "installed lean observation source identity differs"
            )
        return source.observe(group)


class _R06:
    num_envs = 2
    device = torch.device("cpu")
    dtype = torch.float32


def _prepared_epoch():
    owner = E.ActionEpochOwner(num_envs=2, device="cpu", shot_slot_capacity=2)
    owner.activate_reset_genesis(
        selected_mask=torch.ones(2, dtype=torch.bool),
        reset_generation=torch.zeros(2, dtype=torch.int64),
    )
    return owner


def _runtime(env, epoch):
    parts = {
        "r05_runtime": object(),
        "motion": object(),
        "racket": object(),
        "physical_ball": object(),
        "r06_landing_outcome": _R06(),
        "r03_strike_fact": object(),
        "r07_recovery": object(),
    }
    owner = L.ActionBallFullMdpLeanRuntimeOwner(
        env=env,
        runtime_lease=object(),
        epoch_owner=epoch,
        reward_graph=R.LeanActionEpochRewardGraph(epoch_owner=epoch),
        **parts,
    )
    return owner, parts


def _direct_view(env, record, parts):
    slot = record.current_task_slot
    row = slot[:, None]
    value = 1.0
    tensors = {}
    for name, width in O._DIRECT_FIELD_LAYOUT:
        tensors[name] = torch.full((2, width), value)
        value += 1.0
    return O.DirectActionEpochObservationFacts(
        **tensors,
        motion_phase_code=torch.tensor([1, 4], dtype=torch.int64),
        current_task_slot=slot.clone(),
        current_action_uid=torch.gather(record.action_uid, 1, row).squeeze(1),
        current_rng_counter=torch.gather(record.rng_counter, 1, row).squeeze(1),
        transaction_epoch=record.epoch,
        transaction_version=record.version,
        common_step=env.common_step_counter,
        motion_owner=parts["motion"],
        racket_owner=parts["racket"],
        physical_owner=parts["physical_ball"],
        r03_owner=parts["r03_strike_fact"],
        r06_owner=parts["r06_landing_outcome"],
        r07_owner=parts["r07_recovery"],
    )


def test_named_layout_is_compact_and_has_no_legacy_capacity_padding():
    assert O.ACTOR_WIDTH_V1 == 229
    assert O.CRITIC_WIDTH_V1 == 399
    assert [name for name, _ in O.ACTOR_LAYOUT_V1][-7:] == [
        "motion_phase_one_hot",
        "epoch_task_f32",
        "epoch_clock_remaining_s",
        "epoch_phase_one_hot",
        "epoch_task_valid",
        "epoch_selected",
        "epoch_launch_succeeded",
    ]
    assert dict(O.CRITIC_EXTENSION_LAYOUT_V1)["physical_r03_r06_r07_fact_f32"] == 128


def test_missing_exact_direct_runtime_method_holds_before_manager_import(monkeypatch):
    env = _Env()
    runtime, _ = _runtime(env, _prepared_epoch())
    monkeypatch.delattr(
        L.ActionBallFullMdpLeanRuntimeOwner,
        O.DIRECT_VIEW_METHOD,
        raising=True,
    )
    monkeypatch.setattr(O.importlib, "import_module", lambda _name: (_ for _ in ()).throw(AssertionError("must hold first")))
    with pytest.raises(O.LeanObservationConstructionHold, match=O.DIRECT_VIEW_METHOD):
        O.materialize_observation_manager_cfg(env=env, runtime_owner=runtime)


def test_shape_probe_then_semantic_pack_reads_current_public_epoch(monkeypatch):
    env = _Env()
    epoch = _prepared_epoch()
    runtime, parts = _runtime(env, epoch)
    calls = []

    def direct(self, record):
        calls.append(record.version)
        return _direct_view(env, record, parts)

    monkeypatch.setattr(L.ActionBallFullMdpLeanRuntimeOwner, O.DIRECT_VIEW_METHOD, direct, raising=False)

    class Group:
        pass

    class Term:
        def __init__(self, *, func, params):
            self.func, self.params = func, params

    monkeypatch.setattr(
        O.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(ObservationGroupCfg=Group, ObservationTermCfg=Term) if name == "isaaclab.managers" else None,
    )
    bundle = O.materialize_observation_manager_cfg(
        env=env, runtime_owner=runtime
    )
    assert type(bundle) is O.DiagnosticN2ObservationManagerBundle
    env._installed_lean_observation_source = bundle.source
    cfg = bundle.manager_cfg
    policy_term = cfg["policy"].action_epoch
    critic_term = cfg["critic"].action_epoch
    assert policy_term.params == {"group": "policy"}
    assert critic_term.params == {"group": "critic"}
    copied = copy.deepcopy(cfg)
    assert copied["policy"].action_epoch.params == {"group": "policy"}
    assert copied["critic"].action_epoch.params == {"group": "critic"}
    assert policy_term.func(env, **policy_term.params).shape == (2, 229)
    assert critic_term.func(env, **critic_term.params).shape == (2, 399)
    assert calls == []

    env._action_ball_full_mdp_manager_construction_state = "base_managers_complete"
    policy = policy_term.func(env, **policy_term.params)
    critic = critic_term.func(env, **critic_term.params)
    assert calls == [epoch.current().version]
    assert policy.shape == (2, 229) and torch.all(torch.isfinite(policy))
    assert critic.shape == (2, 399) and torch.all(torch.isfinite(critic))
    # First seven direct groups have exactly the published real values.
    cursor = 0
    for expected, (_, width) in enumerate(O._DIRECT_FIELD_LAYOUT, start=1):
        assert torch.all(policy[:, cursor : cursor + width] == float(expected))
        cursor += width
    # Task payload is gathered through the current public slot.
    task_offset = sum(width for _, width in O.ACTOR_LAYOUT_V1[:8])
    expected_task = torch.stack((
        epoch.current().task.task_f32[0, 0],
        epoch.current().task.task_f32[1, 0],
    ))
    assert torch.equal(policy[:, task_offset : task_offset + E.TASK_F32_WIDTH], expected_task)
    # This exact record comes from the current Epoch producer.  It deliberately
    # has no D05-private construction field, while its independent Physical
    # launch fact is a valid false value before launch.
    assert not hasattr(epoch.current(), "construction_admissible")
    assert epoch.current().launch_succeeded[:, 0].tolist() == [False, False]
    assert policy[:, -1].tolist() == [0.0, 0.0]
    assert bundle.source.semantic_publication_count == 1


def test_term_rejects_invalid_group_instance_shadow_and_caller_source(monkeypatch):
    env = _Env()
    epoch = _prepared_epoch()
    runtime, _ = _runtime(env, epoch)

    class Group:
        pass

    class Term:
        def __init__(self, *, func, params):
            self.func, self.params = func, params

    monkeypatch.setattr(
        O.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(
            ObservationGroupCfg=Group, ObservationTermCfg=Term
        )
        if name == "isaaclab.managers"
        else None,
    )
    bundle = O.materialize_observation_manager_cfg(
        env=env, runtime_owner=runtime
    )
    env._installed_lean_observation_source = bundle.source

    with pytest.raises(O.LeanObservationError, match="policy or critic"):
        O._term(env, group="foreign")
    with pytest.raises(TypeError, match="unexpected keyword argument 'source'"):
        O._term(env, group="policy", source=bundle.source)

    env.__dict__[O.ENV_TERM_METHOD] = lambda **_kwargs: torch.zeros((2, 229))
    with pytest.raises(O.LeanObservationConstructionHold, match="shadowed"):
        O._term(env, group="policy")


def test_term_rejects_foreign_source_retained_by_exact_env_resolver(monkeypatch):
    env = _Env()
    foreign_env = _Env()
    runtime, _ = _runtime(foreign_env, _prepared_epoch())

    class Group:
        pass

    class Term:
        def __init__(self, *, func, params):
            self.func, self.params = func, params

    monkeypatch.setattr(
        O.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(
            ObservationGroupCfg=Group, ObservationTermCfg=Term
        )
        if name == "isaaclab.managers"
        else None,
    )
    foreign = O.materialize_observation_manager_cfg(
        env=foreign_env, runtime_owner=runtime
    )
    env._installed_lean_observation_source = foreign.source
    with pytest.raises(
        O.LeanObservationConstructionHold,
        match="source identity differs",
    ):
        O._term(env, group="policy")


def test_nonfinite_direct_fact_fails_without_runtime_zero_fallback(monkeypatch):
    env = _Env()
    epoch = _prepared_epoch()
    runtime, parts = _runtime(env, epoch)

    def bad(self, record):
        del self
        view = _direct_view(env, record, parts)
        return O.DirectActionEpochObservationFacts(
            **{
                **view.__dict__,
                "joint_pos_rel": torch.full((2, 31), float("nan")),
            }
        )

    monkeypatch.setattr(L.ActionBallFullMdpLeanRuntimeOwner, O.DIRECT_VIEW_METHOD, bad, raising=False)
    source = O.LeanActionEpochObservationSource(env=env, runtime_owner=runtime)
    source.observe("policy")
    source.observe("critic")
    env._action_ball_full_mdp_manager_construction_state = "base_managers_complete"
    with pytest.raises(RuntimeError):
        source.observe("policy")
    assert source.semantic_publication_count == 0


def test_source_has_no_superseded_observation_or_zero_prefix_adapter():
    source = (MDP / "action_ball_full_mdp_lean_observation_cfg.py").read_text(encoding="utf-8")
    for marker in (
        "FreshFullMdpObservationOwner",
        "ACTOR_FIXED_WIDTH",
        "CRITIC_FIXED_WIDTH",
        "actor_prefix",
        "critic_prefix",
        "publish_shadow_from_epoch_sources",
        "publish_from_bound_providers",
        "receipt_sha256",
        "source_stamp",
        "numeric_authority",
        "flight_slot_capacity",
        "mailbox_capacity",
    ):
        assert marker not in source
