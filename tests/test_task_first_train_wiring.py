"""Host-only coverage for task-first launch wiring and reward receipts."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import math
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
TASK_FIRST_MDP_DIR = (
    ROOT
    / "hope_training"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
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


def _robot_hit_table():
    return None


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
        face_command_pairing="shared_plus_y",
        racket_body_name="pingpang_red_Link",
        wrist_body_name="right_wrist_yaw_Link",
        mount_offset=(0.21021, 0.032078, 0.032036),
        mount_quat=(1.0, 0.0, 0.0, 0.0),
        mount_normal_axis=1,
        vb_table_near_x=0.5,
        vb_table_surface_z=0.76,
        strike_success_pos_thresh=0.075,
        strike_success_vel_thresh=0.5,
        strike_success_normal_thresh_deg=15.0,
        clean_reference_strike_velocity=True,
        clean_strike_vel_window=2,
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
    broad_regex = (
        r"^(?!left_ankle_roll_Link$)(?!right_ankle_roll_Link$).+$"
    )
    broad_sensor = NS(
        name="contact_forces", body_names=[broad_regex]
    )
    broad_asset = NS(name="robot", body_names=[broad_regex])
    table_prim = "{ENV_REGEX_NS}/TableObstacle"
    return NS(
        obs_mode="hitter_footwork",
        face_command_obs=True,
        station_obs=False,
        physical_ball=False,
        table_obstacle=True,
        table_obstacle_prim=table_prim,
        scene=NS(
            racket_table_contact=NS(
                prim_path=(
                    "{ENV_REGEX_NS}/Robot/right_wrist_yaw_Link"
                ),
                filter_prim_paths_expr=[table_prim],
            ),
            table_obstacle=NS(
                prim_path=table_prim,
                init_state=NS(pos=(1.87, 0.0, 0.735)),
                spawn=NS(
                    size=(2.74, 1.525, 0.05),
                    collision_props=NS(collision_enabled=True),
                ),
            ),
        ),
        terminations=NS(
            base_fell_tilt=NS(
                func="bad_orientation",
                time_out=False,
                params={"limit_angle": 0.7},
            ),
            base_too_low=NS(
                func="root_height_below_minimum",
                time_out=False,
                params={"minimum_height": 0.5},
            ),
            robot_hit_table=NS(
                func="robot_hit_table",
                time_out=False,
                params={
                    "sensor_cfg": broad_sensor,
                    "filtered_sensor_cfg": NS(name="racket_table_contact"),
                    "asset_cfg": broad_asset,
                    "near_x": 0.5,
                    "surface_z": 0.76,
                    "force_threshold": 1.0,
                    "margin": 0.02,
                },
            ),
        ),
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


def _install_task_first_manifest_module(monkeypatch):
    """Expose the dependency-light manifest module without importing Isaac packages."""

    curriculum = _load_by_path(
        "_task_first_train_curriculum_under_test",
        TASK_FIRST_MDP_DIR / "task_first_curriculum.py",
    )
    monkeypatch.setitem(sys.modules, "task_first_curriculum", curriculum)
    manifest = _load_by_path(
        "_task_first_manifest_contract_under_test",
        TASK_FIRST_MDP_DIR / "task_first_manifest.py",
    )
    for name in (
        "whole_body_tracking",
        "whole_body_tracking.tasks",
        "whole_body_tracking.tasks.tracking",
        "whole_body_tracking.tasks.tracking.mdp",
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(
        sys.modules,
        "whole_body_tracking.tasks.tracking.mdp.task_first_manifest",
        manifest,
    )
    return manifest


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
        (
            lambda env: setattr(env, "table_obstacle", False),
            "table_obstacle=true",
        ),
        (
            lambda env: setattr(
                env.terminations, "robot_hit_table", None
            ),
            "robot_hit_table",
        ),
        (
            lambda env: setattr(
                env.commands.racket_target,
                "strike_success_pos_thresh",
                999.0,
            ),
            "strike_success_pos_thresh",
        ),
        (
            lambda env: setattr(
                env.commands.racket_target,
                "clean_reference_strike_velocity",
                False,
            ),
            "clean_reference_strike_velocity",
        ),
        (
            lambda env: setattr(
                env.terminations.base_fell_tilt,
                "func",
                "evil.noop_bad_orientation",
            ),
            "exact function",
        ),
        (
            lambda env: setattr(
                env.terminations.base_too_low, "time_out", True
            ),
            "time_out",
        ),
        (
            lambda env: env.terminations.robot_hit_table.params.__setitem__(
                "force_threshold", 1.0e9
            ),
            "force_threshold=1.0",
        ),
        (
            lambda env: setattr(
                env.scene.racket_table_contact,
                "filter_prim_paths_expr",
                ["{ENV_REGEX_NS}/WrongTable"],
            ),
            "must filter exactly",
        ),
        (
            lambda env: setattr(
                env.scene.table_obstacle.spawn.collision_props,
                "collision_enabled",
                False,
            ),
            "pose/size/collision",
        ),
        (
            lambda env: setattr(
                env.commands.racket_target,
                "mount_offset",
                (0.0, 0.0, 0.0),
            ),
            "physical paddle-site transform",
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
    monkeypatch.setattr(
        train_mod,
        "_load_task_first_manifest_from_racket_cfg",
        lambda _cfg: loaded,
    )
    train_mod._validate_task_first_motion_sources(env_cfg, files)
    files[1] = files[0]
    with pytest.raises(train_mod._OverrideError, match="motion revision mismatch"):
        train_mod._validate_task_first_motion_sources(env_cfg, files)


def test_task_first_hard_contract_binds_manifest_gate_ranges_and_balance(
    train_mod, monkeypatch
):
    _install_task_first_manifest_module(monkeypatch)
    env_cfg = _task_first_env()
    loaded = _fake_loaded(("a", "b"))
    monkeypatch.setattr(
        train_mod,
        "_load_task_first_manifest_from_racket_cfg",
        lambda _cfg: loaded,
    )
    contract = train_mod._task_first_manifest_contract(
        env_cfg.commands.racket_target,
        env_cfg.commands.motion,
        env_cfg,
    )
    assert contract["schema_version"] == 2
    assert contract["manifest_basename"] == "task_first_manifest.json"
    assert contract["manifest_file_sha256"] == "b" * 64
    assert contract["manifest_id"] == "task-first-test"
    assert contract["action_ids"] == ["a", "b"]
    assert contract["action_uids"] == [1000, 1001]
    assert contract["curriculum"]["gate"]["min_attempts"] == 128
    assert contract["ranges"][0]["position_half_extent_m"] == [0.1, 0.2, 0.3]
    assert contract["ranges"][0]["station_center_shift_xy_m"] == [-0.05, 0.0]
    assert contract["success_thresholds"] == {
        "position_m": 0.075,
        "speed_mps": 0.5,
        "face_deg": 15.0,
        "base_m": 0.08,
    }
    assert contract["reference_strike_recipe"]["clean_strike_vel_window"] == 2
    assert contract["motion_sampling"]["balanced_clip_sampling"] is True
    assert contract["motion_sampling"]["balanced_clip_sampling_seed"] == 17
    assert contract["unsafe_evidence"]["termination_terms"] == [
        "base_fell_tilt",
        "base_too_low",
        "robot_hit_table",
    ]


def _agent_cfg(
    *,
    gamma=0.99,
    num_steps_per_env=24,
    max_iterations=1000,
    logger="tensorboard",
    algorithm_extra=None,
):
    algorithm = {
        "class_name": "PPO",
        "learning_rate": 0.001,
        "schedule": "adaptive",
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "gamma": gamma,
        "lam": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "max_grad_norm": 1.0,
    }
    algorithm.update(algorithm_extra or {})
    payload = {
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "save_interval": 100,
        "logger": logger,
        "empirical_normalization": True,
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
        },
        "algorithm": algorithm,
    }
    return NS(to_dict=lambda: payload)


def test_task_first_exact_resume_binds_learning_recipe_not_run_budget(train_mod):
    baseline = train_mod._task_first_agent_recipe(_agent_cfg())
    budget_only = train_mod._task_first_agent_recipe(
        _agent_cfg(max_iterations=9999, logger="wandb")
    )
    changed_gamma = train_mod._task_first_agent_recipe(
        _agent_cfg(gamma=0.98)
    )
    changed_rollout = train_mod._task_first_agent_recipe(
        _agent_cfg(num_steps_per_env=48)
    )

    assert baseline == budget_only
    assert baseline["sha256"] != changed_gamma["sha256"]
    assert baseline["sha256"] != changed_rollout["sha256"]
    assert baseline["recipe"]["runner"]["num_steps_per_env"] == 24
    assert baseline["recipe"]["algorithm"]["gamma"] == 0.99
    assert "max_iterations" not in baseline["recipe"]["runner"]
    assert "logger" not in baseline["recipe"]["runner"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload.__setitem__(
                "num_steps_per_env", True
            ),
            "positive plain integer",
        ),
        (
            lambda payload: payload["algorithm"].__setitem__(
                "gamma", math.nan
            ),
            "non-finite",
        ),
        (
            lambda payload: payload["algorithm"].__setitem__(
                "rnd_cfg", {"weight": 1.0}
            ),
            "does not yet validate state",
        ),
    ),
)
def test_task_first_learning_recipe_rejects_ambiguous_state(
    train_mod, mutate, message
):
    cfg = _agent_cfg()
    payload = cfg.to_dict()
    mutate(payload)
    with pytest.raises(RuntimeError, match=message):
        train_mod._task_first_agent_recipe(cfg)


def test_run_passes_resolved_agent_cfg_into_hard_contract():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_build_training_hard_contract"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords.get("agent_cfg"), ast.Name)
    assert keywords["agent_cfg"].id == "agent_cfg"


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
    with pytest.raises(
        train_mod._OverrideError,
        match=receipt["sha256"],
    ):
        train_mod._build_effective_reward_receipt_for_training(
            env_cfg,
            {},
            require_expected_sha256=True,
        )
    exact = train_mod._build_effective_reward_receipt_for_training(
        env_cfg,
        {"expected_effective_reward_recipe_sha256": receipt["sha256"]},
        require_expected_sha256=True,
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
    module_source = TRAIN_PATH.read_text(encoding="utf-8")
    assert '"effective_reward_recipe": effective_reward_receipt' in hard_source
    assert '"task_first_training": task_first_contract' in hard_source
    assert run_source.index(
        "_build_effective_reward_receipt_for_training"
    ) < run_source.index("gym.make")
    assert "clip0=forehand" not in run_source
    assert "clip1=backhand" not in run_source
    assert "_TASK_FIRST_LOADED_MANIFEST_ATTR" not in module_source
