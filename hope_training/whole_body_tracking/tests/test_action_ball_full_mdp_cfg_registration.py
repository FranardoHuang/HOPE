"""Focused Pod contracts for the fresh full-MDP A/C registration seam.

Registration exposes fourteen lifecycle consumers plus six common dense terms, but never an
executable default RewardManager graph.  Until the factory validates one
numeric authority and replaces the template atomically, construction remains a
truthful HOLD.  Tests here therefore inspect cfg shape only; no Reward callable
or runtime owner is invoked.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import inspect
import io
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "cfg"
REPOSITORY_ROOT = ROOT.parents[1]
SPLIT_ASSET_CONSUMER_SOURCE = (
    ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "config"
    / "agibot_a3"
    / "action_ball_full_mdp_split_asset.py"
)

FRESH_ENTRY_POINT = (
    "whole_body_tracking.tasks.tracking.full_mdp_env:"
    "ActionBallFullMdpManagerBasedRLEnv"
)
FAMILY_CASES = (
    (
        "A",
        "HOPEPingPongActionBallFullMdpA",
        "HOPE-PingPong-ActionBall-FullMdpA-AgibotA3-v0",
        "HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg",
    ),
    (
        "C",
        "HOPEPingPongActionBallFullMdpC",
        "HOPE-PingPong-ActionBall-FullMdpC-AgibotA3-v0",
        "HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg",
    ),
)


def _require_live_isaac_import_surface() -> None:
    """Skip when only an IsaacLab source tree, not a booted Kit runtime, is visible.

    Pod operation venvs can expose ``isaaclab`` through a ``.pth`` file while
    Warp and the ``omni`` extension namespace are intentionally supplied only
    after Kit/AppLauncher boots.  Treating that partial source visibility as a
    runnable Isaac process turns an optional host test into 20+ misleading
    dependency failures.  Real construction is covered by the launcher probe;
    these cfg assertions run only when their complete import surface exists.
    """

    pytest.importorskip("isaaclab")
    pytest.importorskip("warp")
    pytest.importorskip("omni.kit.app")


def _load_split_asset_consumer():
    name = "_test_action_ball_full_mdp_split_asset_consumer"
    spec = importlib.util.spec_from_file_location(
        name, SPLIT_ASSET_CONSUMER_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _independent_fixture_check(root: Path) -> dict[str, object]:
    """Small independent checker used only to exercise the consumer boundary."""

    expected = {
        "model.usd": b"exact derived model",
        "source/urdf/model.urdf": b"exact reviewed urdf",
        "source/meshes/pingpang_red_link.stl": b"exact reviewed red mesh",
        "source/meshes/pingpang_black_link.stl": b"exact reviewed black mesh",
    }
    actual = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "DERIVATION_RECEIPT.json"
    )
    if actual != sorted(expected):
        raise ValueError("actual split-asset inventory differs")
    for relative, content in expected.items():
        path = root / relative
        if path.is_symlink() or path.read_bytes() != content:
            raise ValueError(f"actual split-asset source differs: {relative}")
    # Deliberately adversarial telemetry: the consumer must not read it as an
    # authorization verdict.
    return {"launch_authorized": True, "receipt_matches": True}


def _split_asset_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "a3p0807_split_rubber_diagnostic_v3"
    contents = {
        "model.usd": b"exact derived model",
        "source/urdf/model.urdf": b"exact reviewed urdf",
        "source/meshes/pingpang_red_link.stl": b"exact reviewed red mesh",
        "source/meshes/pingpang_black_link.stl": b"exact reviewed black mesh",
    }
    for relative, content in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def _bind_split_asset_fixture(monkeypatch, consumer, root: Path) -> None:
    monkeypatch.setattr(
        consumer,
        "_load_current_producer_module",
        lambda: SimpleNamespace(check=_independent_fixture_check),
    )
    monkeypatch.setenv("HOPE_AGIBOT_A3_USD_PATH", str(root / "model.usd"))


def _fixture_source_geometry(*, source_urdf: Path, source_mesh_root: Path):
    assert source_urdf.name == "model.urdf"
    assert source_mesh_root.name == "meshes"
    triangle = (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),)
    return (
        {
            "translations": {
                "right_wrist_yaw_link": (0.0, 0.0, 0.0),
                "pingpang_black_link": (0.1, 0.0, 0.0),
                "pingpang_red_link": (0.1, 0.0, 0.0),
                "right_hand_pingpang_link": (0.0, 0.0, 0.0),
            }
        },
        {
            "wrist_shell_collider": triangle,
            "black_rubber_collider": triangle,
            "red_rubber_collider": triangle,
            "racket_handle_collider": triangle,
        },
        {},
    )


def test_split_asset_consumer_has_one_noarg_tracked_verifier():
    consumer = _load_split_asset_consumer()
    assert tuple(
        inspect.signature(
            consumer.require_action_ball_full_mdp_split_asset
        ).parameters
    ) == ()
    assert str(consumer.ACTION_BALL_FULL_MDP_SPLIT_ASSET_ROOT) == (
        "/workspace/franco/runtime_assets/"
        "a3p0807_split_rubber_diagnostic_v3"
    )
    assert consumer.ACTION_BALL_FULL_MDP_SPLIT_ASSET_MODEL == (
        consumer.ACTION_BALL_FULL_MDP_SPLIT_ASSET_ROOT / "model.usd"
    )
    assert consumer._PRODUCER_SOURCE == (
        REPOSITORY_ROOT / "scripts/derive_a3p0807_split_rubber_usd.py"
    )


def test_split_asset_consumer_rejects_relative_selection_and_never_calls_checker(
    monkeypatch,
):
    consumer = _load_split_asset_consumer()
    called = []
    monkeypatch.setattr(
        consumer,
        "_load_current_producer_module",
        lambda: called.append(True),
    )
    monkeypatch.setenv(
        "HOPE_AGIBOT_A3_USD_PATH",
        "relative/private-snapshot/model.usd",
    )
    with pytest.raises(
        consumer.ActionBallFullMdpSplitAssetError,
        match="absolute model.usd",
    ):
        consumer.require_action_ball_full_mdp_split_asset()
    assert called == []


def test_split_asset_consumer_accepts_actual_sources_without_trusting_return(
    tmp_path, monkeypatch
):
    consumer = _load_split_asset_consumer()
    root = _split_asset_fixture(tmp_path)
    _bind_split_asset_fixture(monkeypatch, consumer, root)
    assert consumer.require_action_ball_full_mdp_split_asset() == str(
        root / "model.usd"
    )


def test_split_asset_consumer_accepts_honest_private_snapshot(
    tmp_path, monkeypatch
):
    consumer = _load_split_asset_consumer()
    root = _split_asset_fixture(tmp_path / "fresh-run" / "asset")
    _bind_split_asset_fixture(monkeypatch, consumer, root)
    assert root != consumer.ACTION_BALL_FULL_MDP_SPLIT_ASSET_ROOT
    assert consumer.require_action_ball_full_mdp_split_asset() == str(
        root / "model.usd"
    )


def test_split_asset_expected_geometry_is_noarg_clone_from_enclosed_sources(
    tmp_path, monkeypatch
):
    consumer = _load_split_asset_consumer()
    root = _split_asset_fixture(tmp_path)
    _bind_split_asset_fixture(monkeypatch, consumer, root)
    monkeypatch.setattr(
        consumer,
        "_load_current_producer_module",
        lambda: SimpleNamespace(
            check=_independent_fixture_check,
            _source_geometry=_fixture_source_geometry,
        ),
    )
    assert tuple(
        inspect.signature(
            consumer.action_ball_full_mdp_expected_collider_geometry
        ).parameters
    ) == ()
    view = consumer.action_ball_full_mdp_expected_collider_geometry()
    assert type(view) is consumer.ActionBallFullMdpExpectedColliderGeometry
    assert tuple(mesh.name for mesh in view.meshes) == (
        "wrist_shell_collider",
        "black_rubber_collider",
        "red_rubber_collider",
        "racket_handle_collider",
    )
    assert all(mesh.face_vertex_counts == (3,) for mesh in view.meshes)
    assert all(mesh.face_vertex_indices == (0, 1, 2) for mesh in view.meshes)


@pytest.mark.parametrize("fault", ("partial", "changed_stl", "changed_model"))
def test_split_asset_consumer_rejects_actual_asset_fault(
    tmp_path, monkeypatch, fault: str
):
    consumer = _load_split_asset_consumer()
    root = _split_asset_fixture(tmp_path)
    _bind_split_asset_fixture(monkeypatch, consumer, root)
    if fault == "partial":
        (root / "source/urdf/model.urdf").unlink()
    elif fault == "changed_stl":
        (root / "source/meshes/pingpang_red_link.stl").write_bytes(
            b"changed red mesh"
        )
    else:
        (root / "model.usd").write_bytes(b"changed derived model")
    with pytest.raises(
        consumer.ActionBallFullMdpSplitAssetError,
        match="failed enclosed-source reconstruction",
    ):
        consumer.require_action_ball_full_mdp_split_asset()


@pytest.mark.parametrize("link", ("root", "model"))
def test_split_asset_consumer_rejects_symlink_selection(
    tmp_path, monkeypatch, link: str
):
    consumer = _load_split_asset_consumer()
    real = _split_asset_fixture(tmp_path / "real")
    selected_root = tmp_path / "a3p0807_split_rubber_diagnostic_v3"
    if link == "root":
        selected_root.symlink_to(real, target_is_directory=True)
    else:
        selected_root.mkdir()
        (selected_root / "model.usd").symlink_to(real / "model.usd")
    monkeypatch.setenv(
        "HOPE_AGIBOT_A3_USD_PATH", str(selected_root / "model.usd")
    )
    with pytest.raises(
        consumer.ActionBallFullMdpSplitAssetError,
        match="non-symlink",
    ):
        consumer.require_action_ball_full_mdp_split_asset()


def test_forged_receipt_cannot_cover_changed_actual_model(tmp_path, monkeypatch):
    consumer = _load_split_asset_consumer()
    root = _split_asset_fixture(tmp_path)
    _bind_split_asset_fixture(monkeypatch, consumer, root)
    (root / "DERIVATION_RECEIPT.json").write_text(
        '{"launch_authorized":true,"model_sha256":"forged"}'
    )
    (root / "model.usd").write_bytes(b"changed derived model")
    with pytest.raises(
        consumer.ActionBallFullMdpSplitAssetError,
        match="failed enclosed-source reconstruction",
    ):
        consumer.require_action_ball_full_mdp_split_asset()


def test_receipt_absent_or_forged_does_not_change_actual_source_verdict(
    tmp_path, monkeypatch
):
    consumer = _load_split_asset_consumer()
    root = _split_asset_fixture(tmp_path)
    _bind_split_asset_fixture(monkeypatch, consumer, root)
    assert consumer.require_action_ball_full_mdp_split_asset()
    (root / "DERIVATION_RECEIPT.json").write_text(
        '{"launch_authorized":true,"actual_sources_verified":false}'
    )
    assert consumer.require_action_ball_full_mdp_split_asset()


def test_cfg_requires_actual_asset_before_parent_construction():
    source = (
        ROOT
        / "source"
        / "whole_body_tracking"
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3"
        / "hope_env_cfg.py"
    ).read_text()
    class_start = source.index(
        "class HOPEPingPongActionBallFullMdpAgibotA3EnvCfg("
    )
    post_init = source.index("    def __post_init__(self):", class_start)
    require = source.index(
        "_split_asset.require_action_ball_full_mdp_split_asset()", post_init
    )
    parent = source.index("        super().__post_init__()", post_init)
    assert require < parent


def _compose_task(task_name: str):
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(CFG_DIR.resolve()),
    ):
        return hydra.compose(
            config_name="train",
            overrides=[f"task={task_name}"],
        ).task


def _plain(value):
    omegaconf = pytest.importorskip("omegaconf")
    return omegaconf.OmegaConf.to_container(value, resolve=True)


def _nested_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def _fresh_cfgs(H):
    return (
        H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg(),
        H.HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg(),
    )


def _public_terms(cfg) -> tuple[str, ...]:
    return tuple(
        name
        for name, value in vars(cfg).items()
        if not name.startswith("_") and value is not None
    )


def test_a_c_task_yaml_compose_to_one_common_non_self_authorizing_recipe():
    composed = {}
    for role, task_name, gym_id, _cfg_name in FAMILY_CASES:
        task = _plain(_compose_task(task_name))
        assert task["name"] == task_name
        assert task["gym_task"] == gym_id
        assert task["action_ball_full_mdp_runtime"] is True
        assert task["action_ball_full_mdp_rate_probe"] is False
        assert task["action_ball_full_mdp_family_role"] == role
        assert task["actor_obs_contract"] is None
        assert task["registry_name"] is None
        assert task["registry_name_2"] is None
        assert task["physical_ball"] is False
        assert task["racket"]["target_mode"] == "action_ball_full_mdp"
        assert task["racket"]["action_ball_diagnostic_unauthorized"] is True
        assert task["racket"]["virtual_ball"] is False
        assert task["motion"]["action_ball_diagnostic_split_ready_teacher"] is True
        assert task["motion"]["action_ball_single_stroke_timeout_enabled"] is False

        # A/C role is a launch cross-check, never a YAML-authored payment.
        for key in _nested_keys(task):
            lowered = key.lower()
            assert not (
                "gain" in lowered
                and any(
                    word in lowered
                    for word in ("family", "placement", "treatment")
                )
            )
        common = deepcopy(task)
        for key in ("name", "gym_task", "action_ball_full_mdp_family_role"):
            common.pop(key)
        composed[role] = common

    assert composed["A"] == composed["C"]


def test_gym_specs_resolve_to_custom_env_and_exact_cfg_leaf_types():
    gym = pytest.importorskip("gymnasium")
    _require_live_isaac_import_surface()

    import whole_body_tracking  # noqa: F401
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg

    for _role, _task_name, gym_id, cfg_name in FAMILY_CASES:
        spec = gym.spec(gym_id)
        cfg_type = getattr(hope_env_cfg, cfg_name)
        assert spec.entry_point == FRESH_ENTRY_POINT
        assert spec.entry_point != "isaaclab.envs:ManagerBasedRLEnv"
        assert spec.kwargs["env_cfg_entry_point"] is cfg_type
        assert cfg_type.__name__ == cfg_name


def test_env_cfg_leaves_are_exact_role_projection_with_truthful_hold():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sensors.contact_sensor import ContactSensor

    configs = []
    for cfg_type, role in (
        (H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg, "A"),
        (H.HOPEPingPongActionBallFullMdpCAgibotA3EnvCfg, "C"),
    ):
        cfg = cfg_type()
        configs.append(cfg)
        assert type(cfg) is cfg_type
        assert H.action_ball_full_mdp_family_role(cfg) == role
        assert cfg.obs_mode == "action_ball_full_mdp"
        assert cfg.commands.racket_target.target_mode == "action_ball_full_mdp"
        assert cfg.commands.racket_target.virtual_ball is False
        assert cfg.commands.racket_target.physical_ball is False
        assert cfg.commands.racket_target.shadow_ball is False
        assert cfg.physical_ball is False
        assert cfg.table_obstacle is True
        assert cfg.table_robot_keepout is False
        # IsaacLab's SimulationContext first disables global contact
        # processing.  This already-installed, real ContactSensor is the
        # supported pre-attach consumer that enables it again while
        # InteractiveScene is built; no train-time hot setting or duplicate
        # diagnostic sensor is needed.
        contact_sensor = cfg.scene.contact_forces
        assert type(contact_sensor) is ContactSensorCfg
        assert contact_sensor.class_type is ContactSensor
        assert contact_sensor.prim_path == "{ENV_REGEX_NS}/Robot/.*"
        assert cfg.commands.motion.action_ball_diagnostic_split_ready_teacher is True
        assert cfg.commands.motion.action_ball_single_stroke_timeout_enabled is False
        assert cfg.commands.motion.canonical_ready_mode is True
        assert cfg.commands.motion.balanced_clip_sampling is True
        assert cfg.commands.motion.stand_start_prob == 0.0
        assert cfg.commands.motion.hold_steps_range == (0, 0)
        assert cfg.commands.motion.stand_start_min_hold == 0
        assert cfg.commands.motion.post_swing_min_hold == 0
        assert cfg.commands.motion.post_swing_start_prob == 0.0
        assert cfg.commands.motion.clip_switch_prob == 0.0
        assert cfg.commands.motion.stagger_initial_clock is False
        assert cfg.commands.motion.rsi_skip_settle_frames == 0
        assert cfg.commands.motion.joint_position_range == (0.0, 0.0)
        assert set(cfg.commands.motion.pose_range.values()) == {(0.0, 0.0)}
        assert set(cfg.commands.motion.velocity_range.values()) == {(0.0, 0.0)}
        cadence = cfg.commands.motion.action_ball_continuous_motion_cadence
        assert type(cadence) is dict
        assert cadence["kind"] == (
            H.ACTION_BALL_FULL_MDP_DIAGNOSTIC_MOTION_PROFILE_KIND
        )
        assert set(cadence) == {
            "schema_version",
            "kind",
            "clock_kind",
            "continuous_contract_authority_sha256",
            "recovery_contract_authority_sha256",
            "ready_reference_kind",
            "canonical_sha256",
        }
        assert all(
            len(cadence[name]) == 64
            for name in (
                "continuous_contract_authority_sha256",
                "recovery_contract_authority_sha256",
                "canonical_sha256",
            )
        )
        import action_ball_motion_cadence_device as cadence_module

        parent, receipt, rebuilt = (
            cadence_module.build_action_ball_full_mdp_diagnostic_motion_profile()
        )
        assert type(parent) is (
            cadence_module.DiagnosticMotionParentScheduleAuthority
        )
        assert type(receipt) is cadence_module.DiagnosticMotionProfileReceipt
        assert parent.require_owned_motion_profile(receipt) == cadence == rebuilt
        assert not hasattr(
            cfg, "_action_ball_full_mdp_motion_parent_authority"
        )
        assert not hasattr(cfg, "_action_ball_full_mdp_motion_parent_receipt")
        assert cadence_module.DIAGNOSTIC_UNAUTHORIZED is True
        assert cadence_module.RUNTIME_INTEGRATED is False
        assert cadence_module.LAUNCH_AUTHORIZED is False
        assert cfg.action_ball_full_mdp_runtime_construction_status == "HOLD"
        assert len(cfg.action_ball_full_mdp_scene_spec_sha256) == 64
        assert cfg.action_ball_full_mdp_capacity_receipt_sha256 == ""
        assert cfg.action_ball_full_mdp_scene_capacity == 2
        assert cfg.action_ball_full_mdp_scene_capacity_authority_kind == (
            H.ACTION_BALL_FULL_MDP_DIAGNOSTIC_CAPACITY_AUTHORITY_KIND
        )
        scene_spec = cfg.action_ball_full_mdp_ball_scene_spec
        assert scene_spec.flight_capacity == 2
        assert scene_spec.canonical_sha256 == (
            cfg.action_ball_full_mdp_scene_spec_sha256
        )
        assert scene_spec.capacity_authority_kind == (
            H.ACTION_BALL_FULL_MDP_DIAGNOSTIC_CAPACITY_AUTHORITY_KIND
        )
        assert scene_spec.formal_capacity_receipt_sha256 is None
        assert not hasattr(scene_spec, "capacity_receipt_sha256")
        assert scene_spec.scene_entity_names == (
            "action_ball_flight_ball_000",
            "action_ball_flight_ball_001",
        )
        assert scene_spec.prim_paths == (
            "{ENV_REGEX_NS}/ActionBallFlightBall_000",
            "{ENV_REGEX_NS}/ActionBallFlightBall_001",
        )
        assert getattr(cfg.scene, "pb_ball", None) is None
        assert all(
            getattr(cfg.scene, name, None) is not None
            for name in scene_spec.scene_entity_names
        )
        assert cfg.action_ball_full_mdp_component_roles == (
            "r05_owner",
            "device_r05_owner",
            "motion_owner",
            "racket_owner",
            "r06_owner",
            "physical_owner",
            "r03_owner",
            "r07_owner",
            "ppo_drain_owner",
        )
        assert not hasattr(cfg, "action_ball_full_mdp_placement_gain")
        assert H.action_ball_full_mdp_reward_template_blockers(cfg.rewards) == ()
        assert {
            "fresh Racket command producer remains HOLD",
            "common observation/critic/provider ABI remains R08 HOLD",
            "immutable common A/C motion source and receipt are not launch-bound",
            "nine distinct runtime components are not construction-installed",
            "twenty-term RewardManager numeric authority is not materialized",
        }.issubset(set(cfg.action_ball_full_mdp_construction_blockers))

    left, right = configs
    assert type(left.observations) is type(right.observations)
    assert type(left.rewards) is type(right.rewards)
    assert left.rewards.terms == right.rewards.terms
    assert type(left.terminations) is type(right.terminations)
    assert type(left.commands) is type(right.commands)
    assert (
        left.commands.motion.action_ball_continuous_motion_cadence
        == right.commands.motion.action_ball_continuous_motion_cadence
    )


def test_fresh_a_c_episode_horizon_carries_six_shots_through_retirement():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H

    # Mechanical timing is calculated here from the independently reviewed
    # control schedule.  It is not copied from a source-code receipt or an AST
    # fingerprint: a changed cfg must still satisfy these physical ticks.
    first_reveal_tick = 48
    max_task_close_ticks = 106
    recovery_ticks = 77
    deadline_offset_ticks = 2
    cadence_ticks = (
        max_task_close_ticks + recovery_ticks + deadline_offset_ticks
    )
    accept_reveal_ticks = tuple(
        first_reveal_tick + cadence_ticks * index
        for index in range(6)
    )
    sixth_shot_retirement_tick = accept_reveal_ticks[-1] + cadence_ticks

    assert cadence_ticks == 185
    assert accept_reveal_ticks == (48, 233, 418, 603, 788, 973)
    assert sixth_shot_retirement_tick == 1158

    for cfg in _fresh_cfgs(H):
        step_dt_s = float(cfg.sim.dt) * int(cfg.decimation)
        horizon_ticks = round(float(cfg.episode_length_s) / step_dt_s)
        inherited_ten_second_ticks = round(10.0 / step_dt_s)

        assert step_dt_s == pytest.approx(0.02)
        assert cfg.episode_length_s == pytest.approx(30.0)
        assert horizon_ticks == 1500
        assert horizon_ticks > sixth_shot_retirement_tick
        assert horizon_ticks - sixth_shot_retirement_tick == 95

        # The inherited horizon is a real counterexample: it resets before the
        # second accepted reveal and therefore cannot carry six shots or
        # observe the sixth retirement in one episode.
        assert inherited_ten_second_ticks == 500
        assert inherited_ten_second_ticks < accept_reveal_ticks[1]
        assert inherited_ten_second_ticks < sixth_shot_retirement_tick


def test_fresh_a_c_alone_install_exact_deterministic_robot_reset_event():
    _require_live_isaac_import_surface()
    from isaaclab.managers import EventTermCfg
    from isaaclab.managers import SceneEntityCfg
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking.mdp import events as reset_events

    for cfg in _fresh_cfgs(H):
        assert type(cfg.events) is H.HOPEActionBallFullMdpEventsCfg
        term = cfg.events.action_ball_full_mdp_robot_reset
        public_event_terms = tuple(
            name
            for name, value in vars(cfg.events).items()
            if not name.startswith("_") and type(value) is EventTermCfg
        )
        assert public_event_terms == ("action_ball_full_mdp_robot_reset",)
        assert type(term) is EventTermCfg
        assert (
            term.func
            is reset_events.reset_action_ball_full_mdp_robot_to_physical_ready
        )
        assert term.mode == "reset"
        assert tuple(term.params) == ("asset_cfg",)
        asset_cfg = term.params["asset_cfg"]
        assert type(asset_cfg) is SceneEntityCfg
        assert asset_cfg.name == "robot"
        assert not any(
            type(value) is EventTermCfg
            and value.mode in {"startup", "interval"}
            for value in vars(cfg.events).values()
        )

    legacy_startup_terms = (
        "physics_material",
        "add_joint_default_pos",
        "base_com",
        "randomize_link_mass",
        "randomize_pd_gains",
    )
    for cfg_type in (
        H.HOPEPingPongAgibotA3EnvCfg,
        H.HOPEPingPongHitterAgibotA3EnvCfg,
        H.HOPEPingPongActionBallA211AgibotA3EnvCfg,
        H.HOPEPingPongActionBallC211AgibotA3EnvCfg,
    ):
        cfg = cfg_type()
        assert type(cfg.events) is H.HOPEEventCfg
        assert not hasattr(cfg.events, "action_ball_full_mdp_robot_reset")
        public_event_terms = tuple(
            (name, value.mode)
            for name, value in vars(cfg.events).items()
            if not name.startswith("_") and type(value) is EventTermCfg
        )
        assert public_event_terms == tuple(
            (name, "startup") for name in legacy_startup_terms
        )


def test_official_contact_sensor_enables_processing_before_sim_attach():
    """Pin the three callpoints, not whole dependency files or line numbers."""

    def source(relative: str) -> str:
        expected_root = os.environ.get("HOPE_ISAACLAB_ROOT")
        if expected_root:
            expected = (
                Path(expected_root)
                / "source"
                / "isaaclab"
                / "isaaclab"
                / relative
            )
            assert expected.is_file(), f"exact IsaacLab source is absent: {expected}"
            candidates = [expected.resolve()]
        else:
            candidates = []
            for entry in sys.path:
                path = Path(entry) / "isaaclab" / relative
                if path.is_file():
                    candidates.append(path.resolve())
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            pytest.skip("exact IsaacLab source is unavailable")
        assert len(candidates) == 1, f"ambiguous IsaacLab sources: {candidates!r}"
        return candidates[0].read_text(encoding="utf-8")

    simulation = source("sim/simulation_context.py")
    scene = source("scene/interactive_scene.py")
    contact = source("sensors/contact_sensor/contact_sensor.py")
    environment = source("envs/manager_based_env.py")

    simulation_init = simulation[
        simulation.index("    def __init__(self, cfg:") :
        simulation.index("    def _apply_physics_settings(self):")
    ]
    physics_settings = simulation[
        simulation.index("    def _apply_physics_settings(self):") :
        simulation.index("    def _apply_render_settings_from_cfg(self):")
    ]
    assert simulation_init.index("self._apply_physics_settings()") < simulation_init.index(
        "super().__init__("
    )
    assert (
        'self.carb_settings.set_bool("/physics/disableContactProcessing", True)'
        in physics_settings
    )
    assert scene.index("self._add_entities_from_cfg()") < scene.index(
        "self.clone_environments("
    )
    assert "isinstance(asset_cfg, SensorBaseCfg)" in scene
    assert "asset_cfg.class_type(asset_cfg)" in scene
    assert (
        'set_bool("/physics/disableContactProcessing", False)'
        in contact
    )
    assert environment.index("SimulationContext(self.cfg.sim)") < environment.index(
        "InteractiveScene(self.cfg.scene)"
    ) < environment.index(
        "self.sim.reset()"
    )


def test_fresh_cfg_installs_exact_code_owned_active_n1_motion_catalog():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking.mdp import commands as C

    left, right = _fresh_cfgs(H)
    table = C.load_action_ball_full_mdp_diagnostic_catalog_table()
    assert table.action_order == ("take_061_unit04_bh",)
    assert len(table.action_order) == len(set(table.action_order)) == 1
    assert len(table.motion_files) == len(set(table.motion_files)) == 1
    assert len(table.motion_sha256) == len(set(table.motion_sha256)) == 1
    assert all(
        "/assets/motions/chingmu73_measured_v4_20260803/" in path
        for path in table.motion_files
    )
    assert table.clip_family_per_clip == ("backhand",)
    assert table.mount_normal_sign_per_clip == (1.0,)

    for cfg in (left, right):
        motion = cfg.commands.motion
        racket = cfg.commands.racket_target
        assert C.require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
            motion,
            racket,
            table=table,
        ) is table
        assert motion.motion_file == table.motion_files
        assert tuple(hashlib.sha256(Path(path).read_bytes()).hexdigest()
                     for path in motion.motion_file) == table.motion_sha256
        assert motion.action_ball_full_mdp_diagnostic_catalog == (
            C.ACTION_BALL_FULL_MDP_DIAGNOSTIC_CATALOG_KIND
        )
        assert motion.action_ball_diagnostic_split_ready_teacher is True
        assert motion.action_ball_single_stroke_timeout_enabled is False
        assert motion.canonical_ready_mode is True
        assert racket.clip_names_per_clip == table.action_order
        assert racket.action_ball_diagnostic_unauthorized is True
        assert (
            cfg.actions.joint_pos.pre_apply_guard_diagnostic_compact_evidence
            is False
        )
        assert racket.motion_teacher_racket_source == "measured_channel"

    import action_ball_full_mdp_diagnostic_action_timing as timing

    assert table.manifest_file_sha256 == (
        timing.PINNED_DIAGNOSTIC_MANIFEST_FILE_SHA256
    )
    assert table.manifest_canonical_sha256 == (
        timing.PINNED_DIAGNOSTIC_MANIFEST_CANONICAL_SHA256
    )
    assert timing.DIAGNOSTIC_UNAUTHORIZED is True
    assert timing.FORMAL_ADMISSION is False
    assert timing.RUNTIME_INTEGRATED is False
    assert timing.LAUNCH_AUTHORIZED is False
    with pytest.raises(timing.DiagnosticActionTimingProductionHold):
        timing.construct_production_action_timing_owner()


def test_noncanonical_continuous_cadence_exception_is_catalog_scoped():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking.mdp import commands as C

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    owner = object.__new__(C.MotionCommand)
    owner._debug_vis_handle = None
    owner.cfg = cfg.commands.motion
    owner.canonical_ready_mode = False
    owner.action_ball_single_stroke_timeout_enabled = False
    owner.retiming_active = False
    owner.planner_revision_enabled = False
    owner._event_timing_mode = "disabled"
    owner._action_ball_full_mdp_diagnostic_catalog_table = None
    with pytest.raises(ValueError, match="canonical_ready_mode must be true"):
        C.MotionCommand._configure_action_ball_continuous_motion_cadence(owner)


def test_fresh_catalog_cfg_rejects_caller_override_and_order_or_sign_swap():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking.mdp import commands as C

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    table = C.load_action_ball_full_mdp_diagnostic_catalog_table()
    with pytest.raises(RuntimeError, match="caller-authored motion input"):
        H._attach_action_ball_full_mdp_diagnostic_motion_catalog(cfg)

    cfg.commands.motion.motion_file = ("/wrong/take061.npz",)
    with pytest.raises(ValueError, match="active N=1 diagnostic catalog"):
        C.require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
            cfg.commands.motion,
            cfg.commands.racket_target,
            table=table,
        )

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    cfg.commands.racket_target.clip_names_per_clip = ()
    with pytest.raises(ValueError, match="active N=1 diagnostic catalog"):
        C.require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
            cfg.commands.motion,
            cfg.commands.racket_target,
            table=table,
        )

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    signs = list(cfg.commands.racket_target.mount_normal_sign_per_clip)
    signs[0] *= -1.0
    cfg.commands.racket_target.mount_normal_sign_per_clip = tuple(signs)
    with pytest.raises(ValueError, match="active N=1 diagnostic catalog"):
        C.require_action_ball_full_mdp_diagnostic_catalog_cfg_bindings(
            cfg.commands.motion,
            cfg.commands.racket_target,
            table=table,
        )


@pytest.mark.parametrize("fault", ("bytes_drift", "missing_asset"))
def test_code_owned_catalog_rejects_motion_asset_fault(monkeypatch, fault: str):
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.mdp import commands as C

    table = C.load_action_ball_full_mdp_diagnostic_catalog_table()
    target = Path(table.motion_files[0]).resolve()
    original_read_bytes = Path.read_bytes

    def faulted_read_bytes(path):
        if path.resolve() == target:
            if fault == "bytes_drift":
                return original_read_bytes(path) + b"diagnostic-drift"
            raise FileNotFoundError(target)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", faulted_read_bytes)
    with pytest.raises(ValueError, match="absent or changed|absent|bytes differ"):
        C.load_action_ball_full_mdp_diagnostic_catalog_table()


@pytest.mark.parametrize("device", ("cpu", "cuda:2"))
def test_exact_active_n1_motion_loader_cold_load(device: str):
    _require_live_isaac_import_surface()
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    if device == "cuda:2" and torch.cuda.device_count() < 3:
        pytest.skip("exact Pod GPU2 is unavailable")

    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking.mdp import commands as C

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    table = C.load_action_ball_full_mdp_diagnostic_catalog_table()
    payloads = tuple(Path(path).read_bytes() for path in table.motion_files)
    with np.load(io.BytesIO(payloads[0]), allow_pickle=False) as data:
        articulation_body_names = tuple(str(value) for value in data["body_names"])
    selected_body_names = tuple(cfg.commands.motion.body_names)
    body_indexes = tuple(
        articulation_body_names.index(name) for name in selected_body_names
    )
    loader = C.MotionLoader(
        table.motion_files,
        body_indexes,
        motion_payloads=payloads,
        articulation_body_names=articulation_body_names,
        selected_body_names=selected_body_names,
        device=device,
        allow_legacy_link_origin_velocity=False,
    )
    assert loader.num_segments == 1
    assert loader.kinematics_contract_exact is True
    assert loader.measured_racket_available is True
    assert tuple(
        float(value)
        for value in loader.measured_racket_mount_normal_sign_per_clip
    ) == table.mount_normal_sign_per_clip


def test_fresh_cfg_scene_is_frozen_and_refuses_a_duplicate_pre_env_attach():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import (
        action_ball_full_mdp_ball_scene as S,
    )

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    spec = cfg.action_ball_full_mdp_ball_scene_spec
    with pytest.raises(FrozenInstanceError):
        spec.flight_capacity = 3
    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="already exist",
    ):
        S.attach_action_ball_full_mdp_ball_scene(cfg, spec=spec)

    assert spec.flight_capacity == 2
    assert tuple(
        name
        for name in spec.scene_entity_names
        if getattr(cfg.scene, name, None) is not None
    ) == spec.scene_entity_names


def test_reward_template_matches_exact_shared_reward28_contract():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking import mdp

    reward_cfg = H.HOPEActionBallFullMdpRewardsCfg()
    assert H.action_ball_full_mdp_reward_template_blockers(reward_cfg) == ()
    assert reward_cfg.kind == H.ACTION_BALL_FULL_MDP_REWARD_TEMPLATE_KIND
    assert reward_cfg.status == "HOLD_NUMERIC_AUTHORITY_UNMATERIALIZED"
    assert reward_cfg.numeric_authority_sha256 == ""
    assert reward_cfg.launch_authorized is False
    assert not isinstance(reward_cfg, H.HOPEActionBallRewardsCfg)
    assert _public_terms(reward_cfg) == (
        "schema_version",
        "kind",
        "status",
        "numeric_authority_sha256",
        "launch_authorized",
        "terms",
    )

    terms = reward_cfg.terms
    contract = H._full_mdp_reward_contract
    assert contract.LIFECYCLE_PAYMENT_COUNT == 14
    assert len(terms) == contract.REWARD_TERM_COUNT == 28
    assert tuple(term.manager_name for term in terms) == (
        H.ACTION_BALL_FULL_MDP_REWARD_MANAGER_ORDER
    )
    assert H.ACTION_BALL_FULL_MDP_REWARD_MANAGER_ORDER == contract.MANAGER_NAMES
    lifecycle_count = contract.LIFECYCLE_PAYMENT_COUNT
    common_end = lifecycle_count + len(contract.COMMON_DENSE_SPECS)
    paddle_end = common_end + len(contract.PADDLE_MOTION_PRIOR_SPECS)
    assert tuple(term.payment_consumer for term in terms[:lifecycle_count]) == (
        H._full_mdp_lean_rewards.ORDERED_CONSUMERS
    )
    assert tuple(
        term.payment_consumer for term in terms[lifecycle_count:common_end]
    ) == tuple(
        f"common_dense:{term.manager_name}"
        for term in terms[lifecycle_count:common_end]
    )
    assert tuple(term.payment_consumer for term in terms[common_end:paddle_end]) == tuple(
        f"paddle_motion_prior:{term.manager_name}"
        for term in terms[common_end:paddle_end]
    )
    assert tuple(term.payment_consumer for term in terms[paddle_end:]) == tuple(
        f"regularization:{term.manager_name}" for term in terms[paddle_end:]
    )
    assert tuple(term.func for term in terms) == (
        H._full_mdp_lean_rewards.REWARD_TERM_CALLABLES
    )
    assert tuple(term.owner_role for term in terms) == (
        *("r03_owner" for _ in range(10)),
        "physical_owner",
        "r06_owner",
        "r06_owner",
        "r07_owner",
        *(
            "motion_owner"
            for _ in (
                contract.COMMON_DENSE_SPECS
                + contract.PADDLE_MOTION_PRIOR_SPECS
            )
        ),
        *("regularization_kernel" for _ in contract.REGULARIZATION_SPECS),
    )
    assert tuple(term.manager_weight for term in terms[lifecycle_count:]) == (
        *(spec.manager_weight for spec in contract.COMMON_DENSE_SPECS),
        *(spec.manager_weight for spec in contract.PADDLE_MOTION_PRIOR_SPECS),
        *(spec.manager_weight for spec in contract.REGULARIZATION_SPECS),
    )
    assert terms[lifecycle_count:common_end] == H.ACTION_BALL_FULL_MDP_COMMON_DENSE_TERM_TEMPLATES
    assert terms[common_end:paddle_end] == H.ACTION_BALL_FULL_MDP_PADDLE_MOTION_PRIOR_TERM_TEMPLATES
    assert terms[paddle_end:] == H.ACTION_BALL_FULL_MDP_REGULARIZATION_TERM_TEMPLATES

    for term in terms[:13]:
        assert term.weight_source == H.ACTION_BALL_FULL_MDP_WEIGHT_SOURCE
        assert term.manager_weight is None
        assert term.manager_weight_path.startswith(
            "selected_numeric_parameters.manager_weights."
        )
        assert term.fixed_func_params == ()
    for term in terms[:10]:
        assert term.scale_source == (
            "selected_numeric_parameters.strike_kernel_profiles."
            f"{term.manager_name}.scale"
        )
    recovery = terms[13]
    assert recovery.weight_source == H.ACTION_BALL_FULL_MDP_FIXED_WEIGHT_SOURCE
    assert recovery.manager_weight == 1.0
    assert recovery.manager_weight_path is None
    assert recovery.fixed_func_params == (("manager_weight", 1.0),)
    assert recovery.owner_weight_source.endswith(
        ".recovery.recovery_pose"
    )
    for ordinal, term in enumerate(terms[lifecycle_count:common_end], start=lifecycle_count):
        assert term.weight_source == H.ACTION_BALL_FULL_MDP_COMMON_DENSE_WEIGHT_SOURCE
        assert term.manager_weight > 0.0
        assert term.manager_weight_path is None
        assert term.fixed_func_params[:2] == (
            ("ordinal", ordinal),
            ("command_name", "motion"),
        )
    for ordinal, term in enumerate(terms[common_end:paddle_end], start=common_end):
        assert term.weight_source == H.ACTION_BALL_FULL_MDP_PADDLE_MOTION_PRIOR_WEIGHT_SOURCE
        assert term.fixed_func_params[:2] == (
            ("ordinal", ordinal),
            ("command_name", "racket_target"),
        )
    for ordinal, term in enumerate(terms[paddle_end:], start=paddle_end):
        assert term.weight_source == H.ACTION_BALL_FULL_MDP_REGULARIZATION_WEIGHT_SOURCE
        assert term.fixed_func_params == (("ordinal", ordinal),)
    assert contract.HELD_RACKET_WRIST_BODY_NAME not in H.ACTION_BALL_FULL_MDP_UPPER_NON_WRIST_BODY_NAMES
    assert set(H.ACTION_BALL_FULL_MDP_UPPER_NON_WRIST_BODY_NAMES) < set(H.A3_UPPER_TRACKED)
    for term in terms[lifecycle_count + 2 : common_end]:
        assert dict(term.fixed_func_params)["body_names"] == H.ACTION_BALL_FULL_MDP_UPPER_NON_WRIST_BODY_NAMES
    for term in (
        *terms[lifecycle_count : lifecycle_count + 2],
        *terms[common_end:paddle_end],
        *terms[paddle_end:],
    ):
        assert "body_names" not in dict(term.fixed_func_params)

    forbidden = {
        "death_penalty",
        "table_hit_penalty",
        "virtual_landing",
        "qdes_limit_barrier",
        "joint_limit",
        "base_position",
    }
    assert forbidden.isdisjoint(_public_terms(reward_cfg))
    assert all(term.manager_weight not in (3.0, 11.0, 240.0, 700.0) for term in terms)
    with pytest.raises(RuntimeError, match="construction HOLD: numeric authority"):
        H.require_action_ball_full_mdp_reward_manager_materialized(reward_cfg)


def test_isaac_reward_manager_rejects_unmaterialized_template_before_callable_use():
    _require_live_isaac_import_surface()
    from isaaclab.managers import RewardManager
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H

    class _Sim:
        @staticmethod
        def is_playing() -> bool:
            # Avoid timeline callback setup; this cfg-shape rejection requires
            # neither a simulator nor any scene entity resolution.
            return True

    class _Env:
        sim = _Sim()
        num_envs = 2
        device = "cpu"

    reward_cfg = H.HOPEActionBallFullMdpRewardsCfg()
    with pytest.raises(TypeError, match="not of type RewardTermCfg"):
        RewardManager(reward_cfg, _Env())


def test_placement_shell_is_common_positive_while_treatment_stays_owner_bound():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H

    left, right = _fresh_cfgs(H)
    placement_a = left.rewards.terms[12]
    placement_c = right.rewards.terms[12]
    assert placement_a == placement_c
    assert placement_a.manager_name == "post_contact_placement_guidance"
    assert placement_a.scheduled_for_a is True
    assert placement_a.scheduled_for_c is True
    assert placement_a.manager_weight_must_be_positive is True
    assert placement_a.manager_weight is None
    assert placement_a.manager_weight_path.endswith(".placement")
    assert placement_a.owner_weight_source == "c10_owner_bound_treatment_gain_a1_c0"
    assert not hasattr(left, "action_ball_full_mdp_placement_gain")
    assert not hasattr(right, "action_ball_full_mdp_placement_gain")


def test_termination_cfg_is_exact_five_live_terminal_graph():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking import mdp

    cfg = H.HOPEActionBallFullMdpTerminationsCfg()
    assert not isinstance(cfg, H.HOPEActionBallTerminationsCfg)
    assert _public_terms(cfg) == H.ACTION_BALL_FULL_MDP_TERMINATION_MANAGER_ORDER
    assert tuple(getattr(cfg, name).func for name in _public_terms(cfg)) == (
        mdp.time_out,
        mdp.bad_orientation,
        mdp.root_height_below_minimum,
        mdp.pre_clamp_qdes_forbidden_zone,
        mdp.robot_hit_table,
    )
    assert cfg.time_out.time_out is True
    assert all(
        getattr(cfg, name).time_out is False
        for name in _public_terms(cfg)
        if name != "time_out"
    )
    assert cfg.joint_qdes_forbidden.params["limit_source"] == "joint_pos_limits"
    assert cfg.joint_qdes_forbidden.params["margin_fraction"] == 0.02
    assert cfg.robot_hit_table.params["require_substep_latch"] is False
    assert {
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
        "joint_actual_forbidden",
        "action_ball_single_stroke_timeout",
        "action_ball_strike_fact_publish",
    }.isdisjoint(_public_terms(cfg))

    terminations = importlib.import_module(
        "whole_body_tracking.tasks.tracking.mdp.terminations"
    )
    materialized = (
        terminations.materialize_action_ball_full_mdp_lean_termination_manager_cfg(
            cfg
        )
    )
    assert tuple(materialized) == (
        terminations.ACTION_BALL_FULL_MDP_LEAN_TERMINATION_MANAGER_ORDER
    )
    assert tuple(term.func for term in materialized.values()) == tuple(
        getattr(cfg, name).func for name in _public_terms(cfg)
    )
    assert all(materialized[name] is not getattr(cfg, name) for name in materialized)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("legacy_extra", "public_surface_differs"),
        ("order", "term_differs"),
        ("weight", "term_differs"),
        ("function", "term_differs"),
        ("source", "term_differs"),
    ),
)
def test_reward_template_mutations_fail_closed(mutation: str, expected: str):
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H
    from whole_body_tracking.tasks.tracking import mdp

    cfg = H.HOPEActionBallFullMdpRewardsCfg()
    if mutation == "legacy_extra":
        cfg.virtual_landing = object()
    elif mutation == "order":
        terms = list(cfg.terms)
        terms[0], terms[1] = terms[1], terms[0]
        cfg.terms = tuple(terms)
    elif mutation == "weight":
        terms = list(cfg.terms)
        terms[0] = replace(terms[0], manager_weight=240.0)
        cfg.terms = tuple(terms)
    elif mutation == "function":
        terms = list(cfg.terms)
        terms[0] = replace(terms[0], func=mdp.racket_position_tracking_exp)
        cfg.terms = tuple(terms)
    elif mutation == "source":
        terms = list(cfg.terms)
        terms[0] = replace(terms[0], weight_source="manual_default")
        cfg.terms = tuple(terms)
    else:  # pragma: no cover - parametrization owns the complete mutation set.
        raise AssertionError(mutation)

    blockers = H.action_ball_full_mdp_reward_template_blockers(cfg)
    assert any(expected in blocker for blocker in blockers)
    with pytest.raises(RuntimeError, match="Reward template drift"):
        H.require_action_ball_full_mdp_reward_manager_materialized(cfg)


def test_family_projection_rejects_base_subclass_and_role_rewrite():
    _require_live_isaac_import_surface()
    from whole_body_tracking.tasks.tracking.config.agibot_a3 import hope_env_cfg as H

    base = H.HOPEPingPongActionBallFullMdpAgibotA3EnvCfg()
    with pytest.raises(RuntimeError, match="exact registered EnvCfg type"):
        H.action_ball_full_mdp_family_role(base)

    class ForgedA(H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg):
        pass

    forged = object.__new__(ForgedA)
    with pytest.raises(RuntimeError, match="exact registered EnvCfg type"):
        H.action_ball_full_mdp_family_role(forged)

    cfg = H.HOPEPingPongActionBallFullMdpAAgibotA3EnvCfg()
    cfg.action_ball_full_mdp_family_role = "C"
    with pytest.raises(RuntimeError, match="role was rewritten"):
        H.action_ball_full_mdp_family_role(cfg)


def test_registered_env_constructor_fails_before_simulator_on_real_owner_blocker():
    gym = pytest.importorskip("gymnasium")
    _require_live_isaac_import_surface()

    import whole_body_tracking  # noqa: F401
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking import full_mdp_env as E

    spec = gym.spec(FAMILY_CASES[0][2])
    assert spec.entry_point == FRESH_ENTRY_POINT
    env = object.__new__(E.ActionBallFullMdpManagerBasedRLEnv)
    # The expected error happens before Isaac's base initializer creates this
    # field.  Keep the deliberately partial object safe for ``__del__`` so the
    # fail-closed assertion produces no unrelated unraisable warning.
    env._is_closed = True
    with pytest.raises(
        E.FullMdpPostPhysicsOwnerMissingError,
        match="requires one post-physics owner factory",
    ):
        E.ActionBallFullMdpManagerBasedRLEnv.__init__(env, cfg=object())
