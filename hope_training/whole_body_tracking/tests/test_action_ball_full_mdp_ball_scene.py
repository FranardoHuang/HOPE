from __future__ import annotations

import importlib.util
import inspect
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch


_WBT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _WBT_ROOT / "source" / "whole_body_tracking"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import action_ball_physical_flight_contract as C  # noqa: E402
from test_action_ball_physical_flight_contract import _capacity  # noqa: E402


def _load_scene_module():
    path = (
        _SOURCE_ROOT
        / "whole_body_tracking"
        / "tasks"
        / "tracking"
        / "config"
        / "agibot_a3"
        / "action_ball_full_mdp_ball_scene.py"
    )
    name = "_test_action_ball_full_mdp_ball_scene"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = _load_scene_module()


def _spec(*, cadence: int = 5, horizon: int = 10):
    capacity = _capacity(cadence=cadence, horizon=horizon)
    return capacity, S.build_action_ball_full_mdp_ball_scene_spec(
        capacity_receipt=capacity,
        expected_capacity_receipt_sha256=capacity.canonical_sha256,
        ball_radius_m=0.02,
        ball_mass_kg=0.0034,
    )


def test_scene_spec_derives_k_only_from_frozen_capacity_and_has_no_legacy_alias():
    capacity, spec = _spec(cadence=5, horizon=10)
    assert spec.flight_capacity == 3
    assert spec.flight_capacity == capacity.required_inclusive_flight_capacity
    assert spec.scene_entity_names == (
        "action_ball_flight_ball_000",
        "action_ball_flight_ball_001",
        "action_ball_flight_ball_002",
    )
    assert "pb_ball" not in spec.scene_entity_names
    assert spec.collision_enabled is True
    assert spec.gravity_enabled is True
    assert spec.to_mapping()["canonical_sha256"] == spec.canonical_sha256
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="external pin"):
        S.build_action_ball_full_mdp_ball_scene_spec(
            capacity_receipt=capacity,
            expected_capacity_receipt_sha256="0" * 64,
            ball_radius_m=0.02,
            ball_mass_kg=0.0034,
        )
    with pytest.raises(TypeError):
        S.build_action_ball_full_mdp_ball_scene_spec(  # no capacity default
            expected_capacity_receipt_sha256=capacity.canonical_sha256,
            ball_radius_m=0.02,
            ball_mass_kg=0.0034,
        )


def test_diagnostic_scene_spec_cannot_claim_a_formal_capacity_receipt():
    diagnostic = S.ActionBallFullMdpDiagnosticBallSceneSpec(
        schema_version=S.SCHEMA_VERSION,
        kind=S.DIAGNOSTIC_SCENE_SPEC_KIND,
        capacity_authority_kind="diagnostic_n2_no_save",
        formal_capacity_receipt_sha256=None,
        flight_capacity=2,
        scene_entity_names=(
            "action_ball_flight_ball_000",
            "action_ball_flight_ball_001",
        ),
        prim_paths=(
            "{ENV_REGEX_NS}/ActionBallFlightBall_000",
            "{ENV_REGEX_NS}/ActionBallFlightBall_001",
        ),
        ball_radius_m=0.02,
        ball_mass_kg=0.0034,
        park_position_env_m=S.PARK_POSITION_ENV_M,
        collision_enabled=True,
        gravity_enabled=True,
    )
    assert diagnostic.formal_capacity_receipt_sha256 is None
    assert not hasattr(diagnostic, "capacity_receipt_sha256")
    assert diagnostic.flight_capacity == 2
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="must not claim"):
        replace(diagnostic, formal_capacity_receipt_sha256="0" * 64)
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="exactly K=2"):
        replace(
            diagnostic,
            flight_capacity=3,
            scene_entity_names=diagnostic.scene_entity_names
            + ("action_ball_flight_ball_002",),
            prim_paths=diagnostic.prim_paths
            + ("{ENV_REGEX_NS}/ActionBallFlightBall_002",),
        )

class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RigidObjectCfg(_Cfg):
    InitialStateCfg = _Cfg


def _install_fake_isaac(monkeypatch):
    sim = ModuleType("isaaclab.sim")
    for name in (
        "SphereCfg",
        "RigidBodyPropertiesCfg",
        "MassPropertiesCfg",
        "CollisionPropertiesCfg",
        "PreviewSurfaceCfg",
    ):
        setattr(sim, name, _Cfg)
    assets = ModuleType("isaaclab.assets")
    assets.RigidObjectCfg = _RigidObjectCfg
    package = ModuleType("isaaclab")
    package.sim = sim
    package.assets = assets
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)


def test_attach_materializes_exact_k_collision_enabled_bodies_and_rejects_pb_ball(monkeypatch):
    _, spec = _spec()
    _install_fake_isaac(monkeypatch)
    env_cfg = SimpleNamespace(scene=SimpleNamespace())
    names = S.attach_action_ball_full_mdp_ball_scene(env_cfg, spec=spec)
    assert names == spec.scene_entity_names
    for name, prim_path in zip(names, spec.prim_paths):
        cfg = getattr(env_cfg.scene, name)
        assert cfg.prim_path == prim_path
        assert cfg.spawn.activate_contact_sensors is True
        assert cfg.spawn.collision_props.collision_enabled is True
        assert cfg.spawn.rigid_props.disable_gravity is False
        assert cfg.init_state.pos == S.PARK_POSITION_ENV_M
    assert not hasattr(env_cfg.scene, "pb_ball")

    legacy = SimpleNamespace(scene=SimpleNamespace(pb_ball=object()))
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="legacy"):
        S.attach_action_ball_full_mdp_ball_scene(legacy, spec=spec)


class _Asset:
    def __init__(self, root):
        self.data = SimpleNamespace(root_state_w=root.clone())
        self.pose_writes = 0
        self.velocity_writes = 0

    def write_root_pose_to_sim(self, value, env_ids=None):
        assert env_ids is not None
        self.data.root_state_w[env_ids, :7] = value
        self.pose_writes += 1

    def write_root_velocity_to_sim(self, value, env_ids=None):
        assert env_ids is not None
        self.data.root_state_w[env_ids, 7:] = value
        self.velocity_writes += 1


class _ReplicatedScene(dict):
    def __init__(self, *, num_envs: int):
        super().__init__()
        self.cfg = SimpleNamespace(replicate_physics=True)
        self.env_prim_paths = [
            f"/World/envs/env_{index}" for index in range(num_envs)
        ]


def test_replicated_scene_contract_selects_one_content_source_and_keeps_all_paths():
    scene = _ReplicatedScene(num_envs=4096)
    paths = S._require_replicated_source_scene_paths(scene, num_envs=4096)
    assert len(paths) == 4096
    assert paths[0] == "/World/envs/env_0"
    assert paths[-1] == "/World/envs/env_4095"

    rows = [(object(),) * 7 for _ in range(4096)]
    assert S._replicated_source_stage_row(rows, num_envs=4096) is rows[0]

    source = inspect.getsource(
        S.IsaacLabPhysicalFlightScenePort.install_action_epoch_live_physx_fact_owner
    )
    assert ") in (source_stage_row,):" in source
    assert "for env_row in stage_inventory:" in source


def test_replicated_scene_contract_rejects_heterogeneous_or_wrong_concrete_paths():
    heterogeneous = _ReplicatedScene(num_envs=2)
    heterogeneous.cfg.replicate_physics = False
    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="homogeneous replicated physics",
    ):
        S._require_replicated_source_scene_paths(heterogeneous, num_envs=2)

    wrong_paths = _ReplicatedScene(num_envs=2)
    wrong_paths.env_prim_paths[1] = "/World/envs/foreign"
    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="env prim paths differ",
    ):
        S._require_replicated_source_scene_paths(wrong_paths, num_envs=2)

    with pytest.raises(
        S.ActionBallFullMdpBallSceneError,
        match="inventory width differs",
    ):
        S._replicated_source_stage_row([(object(),) * 7], num_envs=2)


def test_isaac_port_preserves_env_local_state_and_uses_prevalidated_complete_after_image():
    _, spec = _spec(cadence=5, horizon=5)
    assert spec.flight_capacity == 2
    origins = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]], dtype=torch.float32)
    scene = _ReplicatedScene(num_envs=2)
    for name in spec.scene_entity_names:
        root = torch.zeros((2, 13), dtype=torch.float32)
        root[:, :3] = origins + torch.tensor(S.PARK_POSITION_ENV_M)
        root[:, 3] = 1.0
        scene[name] = _Asset(root)
    port = S.IsaacLabPhysicalFlightScenePort(
        scene=scene,
        spec=spec,
        env_origins=origins,
    )
    state = port.read_state_env()
    assert state.shape == (2, 2, 13)
    assert torch.equal(state[..., :3], torch.tensor(S.PARK_POSITION_ENV_M).expand(2, 2, 3))

    candidate = state.clone()
    candidate[1, 0, :3] = torch.tensor([0.3, -0.2, 1.1])
    candidate[1, 0, 7:10] = torch.tensor([4.0, 0.0, -2.0])
    selected = torch.zeros((2, 2), dtype=torch.bool)
    selected[1, 0] = True
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="owned reveal-boundary"):
        port.preflight_write(
            candidate,
            selected,
            reveal_boundary_receipt=None,
        )
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="owned reveal-boundary"):
        port.preflight_write(
            candidate,
            selected,
            reveal_boundary_receipt=object(),
        )
    assert all(asset.pose_writes == 0 for asset in scene.values())
    assert all(asset.velocity_writes == 0 for asset in scene.values())


def test_isaac_port_uses_one_fixed_grid_write_without_dynamic_cuda_selection():
    source = inspect.getsource(S.IsaacLabPhysicalFlightScenePort)
    assert ".nonzero(" not in source
    assert ".item(" not in source
    assert ".tolist(" not in source

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, spec = _spec(cadence=5, horizon=5)
    origins = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]],
        dtype=torch.float32,
        device=device,
    )
    scene = _ReplicatedScene(num_envs=2)
    for name in spec.scene_entity_names:
        root = torch.zeros((2, 13), dtype=torch.float32, device=device)
        root[:, :3] = origins + torch.tensor(
            S.PARK_POSITION_ENV_M,
            dtype=torch.float32,
            device=device,
        )
        root[:, 3] = 1.0
        scene[name] = _Asset(root)
    port = S.IsaacLabPhysicalFlightScenePort(
        scene=scene,
        spec=spec,
        env_origins=origins,
    )
    candidate = port.read_state_env()
    candidate[1, 0, 0] = 0.25
    selected = torch.zeros((2, 2), dtype=torch.bool, device=device)
    selected[1, 0] = True
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="owned reveal-boundary"):
        port.preflight_write(
            candidate,
            selected,
            reveal_boundary_receipt=object(),
        )


def test_isaac_port_rejects_a_writer_that_returns_without_selected_row_readback():
    _, spec = _spec(cadence=5, horizon=4)
    origins = torch.zeros((1, 3), dtype=torch.float32)

    class _NoOpAsset(_Asset):
        def write_root_pose_to_sim(self, value, env_ids=None):
            assert env_ids is not None
            self.pose_writes += 1

        def write_root_velocity_to_sim(self, value, env_ids=None):
            assert env_ids is not None
            self.velocity_writes += 1

    root = torch.zeros((1, 13), dtype=torch.float32)
    root[:, 2] = -20.0
    root[:, 3] = 1.0
    scene = _ReplicatedScene(num_envs=1)
    scene.update(
        {name: _NoOpAsset(root) for name in spec.scene_entity_names}
    )
    port = S.IsaacLabPhysicalFlightScenePort(
        scene=scene,
        spec=spec,
        env_origins=origins,
    )
    candidate = port.read_state_env()
    candidate[0, 0, 0] = 1.0
    with pytest.raises(S.ActionBallFullMdpBallSceneError, match="owned reveal-boundary"):
        port.preflight_write(
            candidate,
            torch.ones((1, 1), dtype=torch.bool),
            reveal_boundary_receipt=object(),
        )


def test_scene_module_is_explicit_hold_not_fake_runtime_go():
    assert S.RUNTIME_INTEGRATED is False
    assert S.POD_FULL_SCENE_VALIDATED is False
    assert S.LAUNCH_AUTHORIZED is False
    assert len(S.INTEGRATION_RESIDUALS) >= 4
    assert S.POSTPHYSICS_FACT_PRODUCERS_BOUND is False
    assert any(
        "N=2 SimulationApp counterexample" in reason
        for reason in S.POSTPHYSICS_CAPTURE_HOLD_REASONS
    )
    assert any(
        "no-save/no-checkpoint" in reason
        for reason in S.POSTPHYSICS_CAPTURE_HOLD_REASONS
    )
    assert S.verify_frozen_physical_flight_contract_source() == S.CONTRACT_SOURCE_SHA256
