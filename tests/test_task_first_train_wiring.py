"""Host-only coverage for task-first launch wiring and reward receipts."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import pathlib
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PATH = (
    ROOT / "hope_training" / "whole_body_tracking" / "scripts" / "train.py"
)
REWARD_RECEIPT_PATH = (
    ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "utils"
    / "effective_reward_recipe.py"
)


def _load_by_path(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_mod():
    if importlib.util.find_spec("hydra") is None:
        hydra = types.ModuleType("hydra")

        def _main(*_args, **_kwargs):
            return lambda function: function

        hydra.main = _main
        sys.modules["hydra"] = hydra
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.ListConfig = list
        omegaconf.OmegaConf = NS(
            resolve=lambda *_args, **_kwargs: None,
            set_struct=lambda *_args, **_kwargs: None,
        )
        sys.modules["omegaconf"] = omegaconf
    return _load_by_path("_task_first_train_under_test", TRAIN_PATH)


class NS(types.SimpleNamespace):
    pass


class FakeObsTerm:
    def __init__(self, func=None, params=None):
        self.func = func
        self.params = dict(params or {})


class FakeGate:
    def as_dict(self):
        return {
            "min_attempts": 128,
            "enter_success_lower_bound": 0.8,
        }


def _fake_action(action_id: str, uid: int, motion_sha256: str = "a" * 64):
    return NS(
        action_id=action_id,
        action_uid=uid,
        motion_sha256=motion_sha256,
        position_half_extent_m=(0.1, 0.2, 0.3),
        speed_delta_mps=1.25,
        face_cone_deg=12.0,
        station_center_shift_xy_m=(-0.05, 0.0),
        base_half_extent_xy_m=(0.15, 0.1),
    )


def _fake_loaded(action_ids=("a", "b"), motion_sha256=None):
    actions = tuple(
        _fake_action(
            action_id,
            1000 + index,
            "a" * 64 if motion_sha256 is None else motion_sha256[index],
        )
        for index, action_id in enumerate(action_ids)
    )
    return NS(
        source_path=pathlib.Path("/review/task_first_manifest.json"),
        file_sha256="b" * 64,
        canonical_sha256="c" * 64,
        manifest=NS(
            manifest_id="task-first-test",
            action_order=tuple(action_ids),
            actions=actions,
            gate=FakeGate(),
        ),
    )


def _task_first_env(action_ids=("a", "b")):
    racket = NS(
        target_mode="task_first",
        task_first_manifest_path="/review/task_first_manifest.json",
        task_first_manifest_sha256="b" * 64,
        task_first_base_success_thresh_m=0.08,
        face_command=True,
        achieved_target_mix_prob=0.0,
        midswing_resample_prob=0.0,
        target_delay_steps=0,
        target_delay_tts_mode="live",
        target_jitter_pos_per_s=0.0,
        target_jitter_vel_per_s=0.0,
        target_noise_white=0.0,
        target_noise_ar1_sigma=0.0,
        target_dropout_prob=0.0,
        target_post_strike_dropout_s=0.0,
        target_bias_per_swing=0.0,
        question_bank="",
        cq_anchor_bank="",
        exam_bank="",
        virtual_ball=False,
        vb_metrics_only=False,
        shadow_ball=False,
        shadow_table=False,
        physical_ball=False,
        planner_revision_enabled=False,
    )
    motion = NS(
        balanced_clip_sampling=True,
        balanced_clip_sampling_seed=17,
        clip_switch_prob=0.0,
        speed_scale_range=(1.0, 1.0),
        speed_scale_per_clip=None,
        event_timing_mode="disabled",
        planner_revision_enabled=False,
    )
    return NS(
        obs_mode="hitter_footwork",
        face_command_obs=True,
        station_obs=False,
        physical_ball=False,
        rewards=NS(),
        commands=NS(racket_target=racket, motion=motion),
        observations=NS(policy=NS()),
    )


def _task_node(action_ids=("a", "b"), contract=None):
    names = list(action_ids)
    return {
        "actor_obs_contract": contract or f"task_first_n{len(names)}",
        "racket": {"clip_names": names},
    }


def _stub_obs_imports(monkeypatch):
    managers = types.ModuleType("isaaclab.managers")
    managers.ObservationTermCfg = FakeObsTerm
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.managers = managers
    mdp = types.ModuleType("whole_body_tracking.tasks.tracking.mdp")
    mdp.racket_target_normal_cmd = object()
    mdp.action_one_hot = object()
    tracking = types.ModuleType("whole_body_tracking.tasks.tracking")
    tracking.mdp = mdp
    tasks = types.ModuleType("whole_body_tracking.tasks")
    tasks.tracking = tracking
    root = types.ModuleType("whole_body_tracking")
    root.tasks = tasks
    for name, module in (
        ("isaaclab", isaaclab),
        ("isaaclab.managers", managers),
        ("whole_body_tracking", root),
        ("whole_body_tracking.tasks", tasks),
        ("whole_body_tracking.tasks.tracking", tracking),
        ("whole_body_tracking.tasks.tracking.mdp", mdp),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return mdp


def _install_fake_manifest_loader(monkeypatch, train_mod, action_ids):
    loaded = _fake_loaded(action_ids)
    monkeypatch.setattr(
        train_mod,
        "_load_task_first_manifest_from_racket_cfg",
        lambda _cfg: loaded,
    )
    return loaded


def test_new_yaml_keys_are_whitelisted_and_consumed(train_mod):
    racket_keys = {
        "task_first_manifest_path",
        "task_first_manifest_sha256",
        "task_first_base_success_thresh_m",
    }
    motion_keys = {"balanced_clip_sampling", "balanced_clip_sampling_seed"}
    assert racket_keys <= set(train_mod._RACKET_KEYS)
    assert motion_keys <= set(train_mod._MOTION_KEYS)

    tree = ast.parse(inspect.getsource(train_mod._apply_task_overrides))
    consumed_attrs = {
        call.args[1].value
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_set_attr"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    }
    assert racket_keys | motion_keys <= consumed_attrs


@pytest.mark.parametrize("action_count", (1, 2, 6))
def test_task_first_appends_face_then_exact_n_action_identity(
    train_mod, monkeypatch, action_count
):
    action_ids = tuple(f"action_{index}" for index in range(action_count))
    env_cfg = _task_first_env(action_ids)
    _install_fake_manifest_loader(monkeypatch, train_mod, action_ids)
    mdp = _stub_obs_imports(monkeypatch)
    applied = []

    train_mod._finalize_task_first_training_cfg(
        env_cfg, _task_node(action_ids), applied
    )

    policy_items = list(vars(env_cfg.observations.policy).items())
    assert [name for name, _ in policy_items] == [
        "racket_target_normal_cmd",
        "action_one_hot",
    ]
    assert policy_items[0][1].func is mdp.racket_target_normal_cmd
    assert policy_items[1][1].func is mdp.action_one_hot
    assert policy_items[1][1].params == {
        "command_name": "racket_target",
        "expected_actions": action_count,
    }
    assert any(f"action_one_hot(+{action_count})" in line for line in applied)


def test_non_task_first_mode_is_strict_noop(train_mod, monkeypatch):
    env_cfg = _task_first_env()
    env_cfg.commands.racket_target.target_mode = "reference_perturbed"
    monkeypatch.setattr(
        train_mod,
        "_load_task_first_manifest_from_racket_cfg",
        lambda _cfg: pytest.fail("non-task-first must not load a manifest"),
    )
    applied = []
    train_mod._finalize_task_first_training_cfg(env_cfg, {}, applied)
    assert vars(env_cfg.observations.policy) == {}
    assert applied == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda env: setattr(env, "obs_mode", "deploy_parity"), "hitter_footwork"),
        (
            lambda env: setattr(
                env.commands.racket_target, "achieved_target_mix_prob", 0.1
            ),
            "achieved_target_mix_prob",
        ),
        (
            lambda env: setattr(env.commands.racket_target, "target_delay_steps", 1),
            "target_delay_steps",
        ),
        (
            lambda env: setattr(
                env.commands.racket_target, "target_noise_ar1_sigma", 0.01
            ),
            "target_noise_ar1_sigma",
        ),
        (
            lambda env: setattr(env.commands.racket_target, "virtual_ball", True),
            "ball-free",
        ),
        (
            lambda env: setattr(env.commands.motion, "balanced_clip_sampling", False),
            "balanced_clip_sampling",
        ),
        (
            lambda env: setattr(env.commands.motion, "clip_switch_prob", 0.1),
            "clip_switch_prob",
        ),
        (
            lambda env: setattr(
                env.commands.motion, "speed_scale_range", (0.8, 1.0)
            ),
            "speed_scale_range",
        ),
        (
            lambda env: setattr(
                env.commands.motion, "event_timing_mode", "post_strike_t1"
            ),
            "event_timing_mode",
        ),
        (
            lambda env: setattr(
                env.commands.motion, "planner_revision_enabled", True
            ),
            "planner_revision",
        ),
    ),
)
def test_task_first_incompatible_modes_fail_before_term_append(
    train_mod, monkeypatch, mutation, message
):
    env_cfg = _task_first_env()
    _install_fake_manifest_loader(monkeypatch, train_mod, ("a", "b"))
    _stub_obs_imports(monkeypatch)
    mutation(env_cfg)
    with pytest.raises(train_mod._OverrideError, match=message):
        train_mod._finalize_task_first_training_cfg(
            env_cfg, _task_node(), []
        )
    assert vars(env_cfg.observations.policy) == {}


def test_task_first_rejects_old_contract_and_preattached_tail(
    train_mod, monkeypatch
):
    env_cfg = _task_first_env()
    _install_fake_manifest_loader(monkeypatch, train_mod, ("a", "b"))
    _stub_obs_imports(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match="task_first_n2"):
        train_mod._finalize_task_first_training_cfg(
            env_cfg, _task_node(contract="hitter_footwork"), []
        )

    env_cfg = _task_first_env()
    env_cfg.observations.policy.racket_target_normal_cmd = FakeObsTerm()
    with pytest.raises(train_mod._OverrideError, match="already attached"):
        train_mod._finalize_task_first_training_cfg(
            env_cfg, _task_node(), []
        )


def test_task_first_rejects_active_ball_outcome_and_absolute_anchor_rewards(
    train_mod, monkeypatch
):
    _install_fake_manifest_loader(monkeypatch, train_mod, ("a", "b"))
    _stub_obs_imports(monkeypatch)

    env_cfg = _task_first_env()
    env_cfg.rewards.virtual_landing = NS(
        func="whole_body_tracking.tasks.tracking.mdp.virtual_landing",
        weight=30.0,
        params={"command_name": "racket_target"},
    )
    with pytest.raises(train_mod._OverrideError, match="virtual_landing"):
        train_mod._finalize_task_first_training_cfg(
            env_cfg, _task_node(), []
        )

    env_cfg = _task_first_env()
    env_cfg.rewards.motion_global_anchor_pos = NS(
        func=(
            "whole_body_tracking.tasks.tracking.mdp."
            "motion_global_anchor_position_error_exp"
        ),
        weight=0.5,
        params={"command_name": "motion"},
    )
    with pytest.raises(train_mod._OverrideError, match="motion_global_anchor_pos"):
        train_mod._finalize_task_first_training_cfg(
            env_cfg, _task_node(), []
        )

    # Explicit zero means Isaac RewardManager skips execution.  A foot landing
    # penalty is contact safety, not a ball-landing outcome, and remains legal.
    env_cfg = _task_first_env()
    env_cfg.rewards.virtual_landing = NS(
        func="whole_body_tracking.tasks.tracking.mdp.virtual_landing",
        weight=0.0,
        params={},
    )
    env_cfg.rewards.foot_soft_landing = NS(
        func="whole_body_tracking.tasks.tracking.mdp.foot_soft_landing",
        weight=-0.1,
        params={},
    )
    train_mod._finalize_task_first_training_cfg(
        env_cfg, _task_node(), []
    )


def test_manifest_order_and_motion_bytes_are_bound(
    train_mod, monkeypatch, tmp_path
):
    env_cfg = _task_first_env()
    loaded = _fake_loaded(("b", "a"))
    monkeypatch.setattr(
        train_mod,
        "_load_task_first_manifest_from_racket_cfg",
        lambda _cfg: loaded,
    )
    _stub_obs_imports(monkeypatch)
    with pytest.raises(train_mod._OverrideError, match="action_order"):
        train_mod._finalize_task_first_training_cfg(
            env_cfg, _task_node(("a", "b")), []
        )

    files = []
    digests = []
    for index in range(2):
        path = tmp_path / f"motion_{index}.npz"
        path.write_bytes(f"motion-{index}".encode())
        files.append(str(path))
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    loaded = _fake_loaded(("a", "b"), digests)
    setattr(
        env_cfg.commands.racket_target,
        train_mod._TASK_FIRST_LOADED_MANIFEST_ATTR,
        loaded,
    )
    train_mod._validate_task_first_motion_sources(env_cfg, files)
    files[1] = files[0]
    with pytest.raises(train_mod._OverrideError, match="motion revision mismatch"):
        train_mod._validate_task_first_motion_sources(env_cfg, files)


def test_task_first_hard_contract_binds_manifest_gate_ranges_and_balance(train_mod):
    env_cfg = _task_first_env()
    loaded = _fake_loaded(("a", "b"))
    setattr(
        env_cfg.commands.racket_target,
        train_mod._TASK_FIRST_LOADED_MANIFEST_ATTR,
        loaded,
    )
    contract = train_mod._task_first_manifest_contract(
        env_cfg.commands.racket_target, env_cfg.commands.motion
    )
    assert contract["manifest_basename"] == "task_first_manifest.json"
    assert contract["manifest_file_sha256"] == "b" * 64
    assert contract["manifest_id"] == "task-first-test"
    assert contract["action_ids"] == ["a", "b"]
    assert contract["action_uids"] == [1000, 1001]
    assert contract["curriculum_gate"]["min_attempts"] == 128
    assert contract["ranges"][0]["position_half_extent_m"] == [0.1, 0.2, 0.3]
    assert contract["ranges"][0]["station_center_shift_xy_m"] == [-0.05, 0.0]
    assert contract["motion_balanced_clip_sampling"] is True
    assert contract["motion_balanced_clip_sampling_seed"] == 17


def _reward_func():
    return None


@dataclasses.dataclass
class _SceneEntityCfgHostMirror:
    """Isaac ``@configclass`` is a dataclass; host fallback uses its field shape."""

    name: str
    joint_names: list[str] | None = None
    body_names: list[str] | None = None
    joint_ids: slice = dataclasses.field(default_factory=lambda: slice(None))


def _scene_entity_cfg():
    try:
        from isaaclab.managers import SceneEntityCfg
    except (ImportError, ModuleNotFoundError):
        return _SceneEntityCfgHostMirror(
            "robot", joint_names=[".*_ankle.*"], body_names=["left_foot"]
        )
    return SceneEntityCfg(
        "robot", joint_names=[".*_ankle.*"], body_names=["left_foot"]
    )


def _install_reward_module(monkeypatch):
    module = _load_by_path(
        "whole_body_tracking.utils.effective_reward_recipe",
        REWARD_RECEIPT_PATH,
    )
    utils = types.ModuleType("whole_body_tracking.utils")
    utils.__path__ = []
    root = types.ModuleType("whole_body_tracking")
    root.__path__ = []
    root.utils = utils
    monkeypatch.setitem(sys.modules, "whole_body_tracking", root)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.utils", utils)
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.utils.effective_reward_recipe",
        module,
    )
    return module


def test_reward_receipt_serializes_isaac_scene_entity_configclass_shape(
    train_mod, monkeypatch
):
    _install_reward_module(monkeypatch)
    selector = _scene_entity_cfg()
    assert dataclasses.is_dataclass(selector)
    env_cfg = NS(
        rewards=NS(
            foot_contact=NS(
                func=_reward_func,
                weight=-0.5,
                params={"asset_cfg": selector},
            )
        )
    )
    receipt = train_mod._build_effective_reward_receipt_for_training(
        env_cfg, {}
    )
    serialized = receipt["terms"][0]["params"]["asset_cfg"]
    assert serialized["__config_type__"].endswith(type(selector).__qualname__)
    assert serialized["fields"]["name"] == "robot"
    assert serialized["fields"]["joint_names"] == [".*_ankle.*"]
    assert serialized["fields"]["joint_ids"] == {
        "__slice__": [None, None, None]
    }


def test_effective_reward_expected_sha_and_persisted_contract_binding(
    train_mod, monkeypatch, tmp_path
):
    _install_reward_module(monkeypatch)
    env_cfg = NS(
        rewards=NS(
            racket_position=NS(
                func=_reward_func,
                weight=4.0,
                params={"std": 0.075, "command_name": "racket_target"},
            )
        )
    )
    receipt = train_mod._build_effective_reward_receipt_for_training(
        env_cfg, {}
    )
    assert receipt["terms"][0]["weight"] == 4.0
    exact = train_mod._build_effective_reward_receipt_for_training(
        env_cfg,
        {"expected_effective_reward_recipe_sha256": receipt["sha256"]},
    )
    assert exact == receipt
    with pytest.raises(ValueError, match="mismatch"):
        train_mod._build_effective_reward_receipt_for_training(
            env_cfg,
            {"expected_effective_reward_recipe_sha256": "0" * 64},
        )

    path = tmp_path / "params" / "effective_reward_recipe.json"
    train_mod._write_effective_reward_receipt(
        path, receipt, {"effective_reward_recipe": receipt}
    )
    assert path.is_file()
    with pytest.raises(RuntimeError, match="embedded hard contract"):
        train_mod._write_effective_reward_receipt(
            path, receipt, {"effective_reward_recipe": {"sha256": "bad"}}
        )


def test_source_binds_receipt_and_has_no_two_clip_semantic_log(train_mod):
    hard_source = inspect.getsource(train_mod._build_training_hard_contract)
    run_source = inspect.getsource(train_mod._run)
    assert '"effective_reward_recipe": effective_reward_receipt' in hard_source
    assert '"task_first_training": task_first_contract' in hard_source
    assert run_source.index(
        "_build_effective_reward_receipt_for_training"
    ) < run_source.index("gym.make")
    assert "clip0=forehand" not in run_source
    assert "clip1=backhand" not in run_source
